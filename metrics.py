import torch


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        pred = pred.view(-1) if pred.dim() != 1 else pred
        target = target.view(-1) if target.dim() != 1 else target
        ## pred & target are flatten (B*H*W) and contain class index of each pixel in each sample in the current batch
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n) & (pred >= 0) ## filter out invalid classes (-1 or >=num_classes)
            inds = n * target[k].to(torch.int64) + pred[k]
            ## 'inds' is a vector of length B*H*W. Each of its entry specifies index in a flatten matrix of size (n x n)
            ## whose rows indiate target/gt classes and columns indicate predicted classes of all pixels of all samples
            ## in current batch. e.g. if n(=num_classes) is 21, an entry of 68(=21*3 + 5) indicates the pixel true class
            ## is 3 but its predicted class is 5
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            ## bincount counts number of occurences of indies in 'inds' and reshape it to (n x n) to form the above
            ## mentioned matrix which is the confusion matrix of current batch

    def get_acc(self, h):
        acc = torch.diag(h).sum() / h.sum()
        ## torch.diag(h) entries specify number of correctly classified samples (pixels) of each class and their sum()
        ## is total number of all correctly classified samples. Apparently h.sum() specify total number of samples.
        return acc.item()

    def get_iou(self, h):
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-15)
        ## torch.diag(h) entries, as stated before, specify number of correctly classified pixels of each class or in
        ## other word, number of pixels in intersection of target and prediction for each class
        ## h.sum(1) entries are the number of pixels predicted as each class
        ## h.sum(0) entries are the number of pixels in each class (target or ground-truth)
        ## (h.sum(1) + h.sum(0) - torch.diag(h)) is the union of pixels either predicted or truly are in each class
        ## 'iu' entries are IoU of each class
        return iu
        # return torch.mean(iu).item()

    def get_metrics(self, _16trainids=None, _13trainids=None):
        if self.mat is None:
            return {'19cls': torch.Tensor([0])}, {'19cls': 0}
        acc = dict()
        iou = dict()

        h = self.mat.float()
        acc['19cls'] = self.get_acc(h)
        iou['19cls'] = self.get_iou(h)
        if _16trainids is not None:
            # IndexError: tensors used as indices must be long, byte or bool tensors
            _16trainids = torch.Tensor(_16trainids).type(torch.long)
            h16 = h[_16trainids[:, None], _16trainids]
            acc['16cls'] = self.get_acc(h16)
            iou['16cls'] = self.get_iou(h16)
        if _13trainids is not None:
            _13trainids = torch.Tensor(_13trainids).type(torch.long)
            h13 = h[_13trainids[:, None], _13trainids]
            acc['13cls'] = self.get_acc(h13)
            iou['13cls'] = self.get_iou(h13)

        return iou, acc
