from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

__all__ = ["SoftBCEWithLogitsLoss"]


class SoftBCEWithLogitsLoss(nn.Module):
    """
        Drop-in replacement for nn.BCEWithLogitsLoss with few additions:
        - Support of ignore_index value
        - Support of label smoothing
    """

    __constants__ = ["weight", "pos_weight", "reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self, weight=None, ignore_index: Optional[int] = -100, reduction="mean", smooth_factor=None, pos_weight=None
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.smooth_factor is not None:
            soft_targets = (1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)
        else:
            soft_targets = target

        loss = F.binary_cross_entropy_with_logits(
            input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        if self.ignore_index is not None:
            not_ignored_mask = (target != self.ignore_index)
            size = not_ignored_mask.sum()
            if size == 0:
                # If there are zero elements, loss is zero
                return 0
            loss *= not_ignored_mask.to(loss.dtype)
        else:
            size = loss.numel()

        if self.reduction == "mean":
            loss = loss.sum() / size

        if self.reduction == "sum":
            loss = loss.sum()

        return loss
