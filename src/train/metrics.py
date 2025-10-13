import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def iou_binary(pred, target, thresh=0.5, eps=1e-6):
    pb = (pred > thresh).float()
    tb = (target > thresh).float()
    inter = (pb * tb).sum().item()
    union = (pb + tb).clamp(max=1).sum().item()
    return inter / (union + eps)


def precision_recall_f1(pred, target, thresh=0.5, eps=1e-6):
    pb = (pred > thresh).float()
    tb = (target > thresh).float()
    tp = (pb * tb).sum().item()
    fp = ((pb == 1) & (tb == 0)).sum().item()
    fn = ((pb == 0) & (tb == 1)).sum().item()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return prec, rec, f1
