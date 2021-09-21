from datasets.transforms import ToLabel, ToTensor, Normalize, ReLabel, Compose
from datasets.cv2_aug_transforms import RandomResize, RandomCrop, RandomHFlip, RandomBrightness


def get_train_transforms():
    return {
        'transforms': Compose([
            RandomResize(resize_ratio=1.0, method='random',
                         scale_range=[0.5, 2.0], aspect_range=[0.9, 1.1]),
            RandomCrop(crop_ratio=1.0, method='random',
                       crop_size=[1024, 512], allow_outside_center=False),
            RandomHFlip(flip_ratio=0.5, swap_pair=[]),
            RandomBrightness(brightness_ratio=1.0, shift_value=10)
        ]),
        'img_transform': Compose([
            ToTensor(),
            Normalize(
                div_value=255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'label_transform': Compose([
            ToLabel(),
            ReLabel(255, -1)
        ])
    }


def get_val_transforms():
    return {
        'transforms': None,
        'img_transform': Compose([
            ToTensor(),
            Normalize(
                div_value=255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'label_transform': Compose([
            ToLabel(),
            ReLabel(255, -1)
        ])
    }


def get_transforms():
    return get_train_transforms(), get_val_transforms()
