from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", smooth_factor=None, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        num_classes = input.size(1)

        log_prb = F.log_softmax(input, dim=1)
        ce_loss = F.nll_loss(log_prb, target.long(), ignore_index=self.ignore_index, reduction=self.reduction)
        if self.smooth_factor is None:
            return ce_loss

        if self.ignore_index is not None:
            not_ignored_mask = (target != self.ignore_index).to(input.dtype)
            log_prb *= not_ignored_mask.unsqueeze(dim=1)

        if self.reduction == "sum":
            smooth_loss = -log_prb.sum()
        else:
            smooth_loss = -log_prb.sum(dim=1)  # We divide by that size at the return line so sum and not mean
            if self.reduction == "mean":
                smooth_loss = smooth_loss.mean()

        return self.smooth_factor * smooth_loss / num_classes + (1 - self.smooth_factor) * ce_loss
