import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class DiceLoss(nn.Module):
    """Soft Dice loss on [0,1] maps. Interprets values as probabilities."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred,target: (B,1,H,W)
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        inter = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    """Placeholder boundary loss: encourages sharper transitions using gradient magnitude.
    This is a simplified surrogate; can (should) replace with preferred boundary-aligned loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        def grad_mag(x):
            # Sobel-like filters
            gx = F.pad(x, (0,1,0,0))[:, :, :, 1:] - F.pad(x, (1,0,0,0))[:, :, :, :-1]
            gy = F.pad(x, (0,0,0,1))[:, :, 1:, :] - F.pad(x, (0,0,1,0))[:, :, :-1, :]
            return torch.sqrt(gx**2 + gy**2 + 1e-6)
        gp = grad_mag(pred)
        gt = grad_mag(target)
        return F.l1_loss(gp, gt)
