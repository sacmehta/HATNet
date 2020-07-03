import torch
from torch.nn import functional as F


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_f1(y_pred: torch.Tensor, y_true: torch.Tensor, n_classes=4, epsilon=1e-7, is_one_hot=False):
    if is_one_hot:
        # B x C
        assert y_pred.dim() == y_true.dim()
    else:
        assert len(y_pred.size()) == 2 # B x C
        assert len(y_true.size()) == 1 # B

    with torch.no_grad():
        y_true = y_true.to(torch.float32) if is_one_hot else F.one_hot(y_true.to(torch.int64), n_classes).to(torch.float32)
        y_pred = y_pred.argmax(dim=1)
        y_pred = F.one_hot(y_pred.to(torch.int64), n_classes).to(torch.float32)

        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return torch.mean(f1) * 100