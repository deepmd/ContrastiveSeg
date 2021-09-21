import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from util import set_up_logger, log_parameters, save_checkpoint
from datasets import get_transforms
from datasets.collate import collate
from datasets.cityscapes import Cityscapes
from networks.hrnet import HRNet_W48_CONTRAST
from optim_scheduler import get_optim_scheduler
from losses import ContrastCELoss
from util import AverageMeter
from metrics import ConfMatrix
from networks.sync_batchnorm import convert_model, DataParallelWithCallback


def parse_options():
    parser = argparse.ArgumentParser('arguments for training')

    # model, dataset
    parser.add_argument('--data_folder', type=str, default=None, help='path to dataset')
    parser.add_argument('--model', type=str, default='HRNet-W48-Contrast', choices=['HRNet-W48-Contrast'])
    parser.add_argument('--proj_dim', type=int, default=256, help='num of channels in output of projection head')
    parser.add_argument('--num_classes', type=int, default=19, help='number of classes.')
    parser.add_argument('--pretrained', type=str, default=None, help='path to pretrained weights.')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr_policy', type=str, default='lambda_poly', help='scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--loss_weight", type=float, default=0.1, help="the weight is used for balancing losses.")

    # contrastive loss
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature in contrastive loss.")
    parser.add_argument("--base_temperature", type=float, default=0.07, help="base temperature in contrastive loss.")
    parser.add_argument('--max_samples', type=int, default=1024, help='max samples')
    parser.add_argument('--max_views', type=int, default=100, help='max views')

    # train settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--max_iters', type=int, default=40000, help='max iterations for training.')
    parser.add_argument('--contrast_warmup_iters', type=int, default=5000, help='warmup iterations for training.')

    # # other setting
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--eval_freq', type=int, default=1000, help='evaluation model on validation set frequency')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")

    opt = parser.parse_args()

    train_data_transformer = dict(size_mode="fix_size", input_size=[1024, 512],
                                  align_method="only_pad", pad_mode="random")
    val_data_transformer = dict(size_mode="fix_size", input_size=[2048, 1024],
                                align_method="only_pad")
    opt.data_transformer = dict(train=train_data_transformer, val=val_data_transformer)

    opt.ce_weights = [0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                      1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                      1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
    opt.ce_ignore_index = -1

    opt.model_name = f"cityscapes_model_{opt.model}_{opt.optimizer}_syncbn_{opt.syncBN}" + \
                     f"_lr_{opt.learning_rate}_bsz_{opt.batch_size}_loss_CE-Contrast_trial_{opt.trial}"

    save_path = os.path.join("./save", opt.model_name)

    opt.tb_folder = os.path.join(save_path, "tensorboard")
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(save_path, "models")
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(save_path, "logs")
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_loader(opt):
    logger = opt.logger
    train_transforms, val_transforms = get_transforms()

    # train data
    train_dataset = Cityscapes(
        root=opt.data_folder,
        split='train',
        transforms=train_transforms['transforms'],
        img_transform=train_transforms['img_transform'],
        label_transform=train_transforms['label_transform'],
        debug=opt.debug
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda *args: collate(*args, opt.data_transformer['train'])
    )

    # validation data
    val_dataset = Cityscapes(
        root=opt.data_folder,
        split='val',
        transforms=val_transforms['transforms'],
        img_transform=val_transforms['img_transform'],
        label_transform=val_transforms['label_transform'],
        debug=opt.debug
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda *args: collate(*args, opt.data_transformer['val'])
    )

    logger.info(f"Summary of the data:")
    logger.info(f"Number of images in training set = {len(train_dataset)}")
    logger.info(f"Number of images in validation set = {len(val_dataset)}")

    return train_loader, val_loader


def set_model(opt):
    logger = opt.logger

    model = None
    if opt.model == 'HRNet-W48-Contrast':
        model = HRNet_W48_CONTRAST(num_classes=opt.num_classes, bn_type='torchbn', proj_dim=opt.proj_dim)
    else:
        logger.info(f'model {opt.model} is not implemented.')

    if opt.pretrained is not None:
        logger.info('Loading pretrained model:{}'.format(opt.pretrained))
        pretrained_dict = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
        model_backbone_dict = model.backbone.state_dict()
        load_backbone_dict = {k: v for k, v in pretrained_dict.items() if k in model_backbone_dict.keys()}
        model_backbone_dict.update(load_backbone_dict)
        missing_keys, unexpected_keys = model.backbone.load_state_dict(model_backbone_dict)
        logger.info('Missing keys: {}'.format(missing_keys))
        logger.info('Unexpected keys: {}'.format(unexpected_keys))

    criterion = ContrastCELoss(opt.ce_weights, opt.ce_ignore_index, opt.loss_weight,
                               opt.max_samples, opt.max_views, opt.temperature, opt.base_temperature)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            if opt.syncBN:
                model = convert_model(model)
                model = DataParallelWithCallback(model)
                logger.info('model is using synchronized batch normalization')
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    optimizer, scheduler = get_optim_scheduler(model, opt)

    return model, criterion, optimizer, scheduler


def train_validate(model, criterion, data_loaders, optimizer, scheduler, opt):
    model.train()
    criterion.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    contrast_losses = AverageMeter()

    end = time.time()
    for data_dict in data_loaders['train']:
        data_time.update(time.time() - end)

        bsz = data_dict['img'].size(0)
        images = data_dict['img']
        targets = data_dict['labelmap']
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # compute loss
        logits = model(images)
        with_embed = True if opt.current_iter >= opt.contrast_warmup_iters else False
        loss, partial_losses = criterion(logits, targets, with_embed=with_embed)

        with torch.no_grad():
            losses.update(loss.item(), bsz)
            ce_losses.update(partial_losses['ce'], bsz)
            contrast_losses.update(partial_losses['contrast'], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(opt.current_iter)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # updating num of iterations till now
        opt.current_iter += 1

        # add info to tensorboard
        opt.tb_logger.add_scalar('train/total_loss', loss, opt.current_iter)
        opt.tb_logger.add_scalar('train/ce_loss', partial_losses['ce'], opt.current_iter)
        opt.tb_logger.add_scalar('train/contrast_loss', partial_losses['contrast'], opt.current_iter)
        opt.tb_logger.add_scalar('train/learning_rate_groups0', optimizer.param_groups[0]['lr'], opt.current_iter)
        opt.tb_logger.add_scalar('train/learning_rate_groups1', optimizer.param_groups[1]['lr'], opt.current_iter)

        # print info
        if opt.current_iter % opt.print_freq == 0:
            opt.logger.info(f"[Train] [Epoch {opt.current_epoch}] " +
                            f"[Iteration {opt.current_iter}/{opt.max_iters}] " +
                            f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t" +
                            f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t" +
                            f"Total Loss: {losses.val:.3f} ({losses.avg:.3f}) " +
                            f"CE Loss: {ce_losses.val:.3f} ({ce_losses.avg:.3f}) " +
                            f"Contrast Loss: {contrast_losses.val:.3f} ({contrast_losses.avg:.3f})\t" +
                            "Learning rate: {}".format([param_group['lr'] for param_group in optimizer.param_groups]))

        if opt.current_iter % opt.eval_freq == 0 or opt.current_iter % opt.max_iters == 0:
            class_ious, pixel_acc, val_loss_avg = validate(model, data_loaders['val'], criterion, opt)
            opt.logger.info(f"[Validation] [Epoch {opt.current_epoch}] "
                            f"[Iteration {opt.current_iter}]\t" +
                            f"Loss: {val_loss_avg:.3f}, " +
                            f"Mean IOU: {torch.mean(class_ious).item():.3f}, Pixel ACC: {pixel_acc:.3f}")
            if opt.current_iter % opt.max_iters == 0:
                opt.logger.info(f" Class IOU: <{'.'.join(f'{idx}: {iou}' for idx, iou in enumerate(class_ious))}>")

            # add info to tensorboard
            opt.tb_logger.add_scalar('val/mIOU', torch.mean(class_ious), opt.current_iter)
            opt.tb_logger.add_scalar('val/pixel_acc', pixel_acc, opt.current_iter)

            # save model
            save_file = os.path.join(
                opt.save_folder, 'ckpt_iteration_{iteration}_{miou}.pth'.format(iteration=opt.current_iter,
                                                                                miou=torch.mean(class_ious).item()))
            if opt.current_iter % opt.max_iters == 0:
                save_file = os.path.join(opt.save_folder, 'last_{miou}.pth'.format(miou=torch.mean(class_ious).item()))
            save_checkpoint(model, optimizer, opt, opt.current_iter, save_file)

            # changing the phase of the model to train
            model.train()
            criterion.train()

        if opt.current_iter % opt.max_iters == 0:
            break

    opt.current_epoch += 1

    return losses.avg, ce_losses.avg, contrast_losses.avg


def validate(model, val_loader, criterion, opt):
    model.eval()
    criterion.eval()

    losses = AverageMeter()
    metrics = ConfMatrix(opt.num_classes)

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(val_loader)):
            bsz = data_dict['img'].size(0)
            images = data_dict['img']
            targets = data_dict['labelmap']

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            logits = model(images)
            loss, _ = criterion(logits, targets)

            logits_seg = logits['seg']
            if list(logits_seg.shape[2:]) != list(targets.shape[1:]):
                logits_seg = F.interpolate(logits_seg, [targets.size(1), targets.size(2)],
                                           mode='bicubic', align_corners=True)

            preds = torch.argmax(logits_seg, dim=1)
            metrics.update(preds, targets)

        losses.update(loss.item(), bsz)
        class_ious_dict, pixel_acc_dict = metrics.get_metrics()
        class_ious = class_ious_dict['19cls']
        pixel_acc = pixel_acc_dict['19cls']

    return class_ious, pixel_acc, losses.avg


def main():
    opt = parse_options()

    # Set deterministic training for reproducibility
    manual_seed = 304
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    # logger
    logger = set_up_logger(logs_path=opt.log_folder)
    log_parameters(opt, logger)
    opt.logger = logger

    # tensorboard
    tb_logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=30)
    opt.tb_logger = tb_logger

    # build data loader
    train_loader, val_loader = set_loader(opt)
    data_loaders = dict(train=train_loader, val=val_loader)

    # build model, criterion and optimizer and scheduler
    model, criterion, optimizer, scheduler = set_model(opt)

    # training routine
    opt.current_epoch = 0
    opt.current_iter = 0
    while opt.current_iter < opt.max_iters:
        # opt.current_iter and opt.current_epoch are updated in the train function.
        time1 = time.time()
        loss, ce_loss, contrast_loss = train_validate(model, criterion, data_loaders, optimizer, scheduler, opt)
        time2 = time.time()
        logger.info(f"[End Epoch {opt.current_epoch-1}] Train Time: {(time2 - time1):0.2f}, " +
                    f"Loss: {loss:06.3f} (CE: {ce_loss:06.3f}, Contrast: {contrast_loss:06.3f})")
        opt.tb_logger.add_scalar('train/total_loss_avg', loss, opt.current_epoch)
        opt.tb_logger.add_scalar('train/ce_loss_avg', ce_loss, opt.current_epoch)
        opt.tb_logger.add_scalar('train/contrast_loss_avg', contrast_loss, opt.current_epoch)


if __name__ == '__main__':
    main()