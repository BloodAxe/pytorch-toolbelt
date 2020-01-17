from typing import Optional

import torch.nn.functional as F
from torch import nn

__all__ = ["BCELoss", "SoftBCELoss"]


class BCELoss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = -100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, label_input, label_target):
        if self.ignore_index is not None:
            not_ignored_mask = (label_target != self.ignore_index).float()

        loss = F.binary_cross_entropy_with_logits(label_input, label_target, reduction="none")
        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftBCELoss(nn.Module):
    def __init__(self, smooth_factor=0, ignore_index: Optional[int] = -100, reduction="mean"):
        super().__init__()
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, label_input, label_target):
        if self.ignore_index is not None:
            not_ignored_mask = (label_target != self.ignore_index).float()

        label_target = (1 - label_target) * self.smooth_factor + label_target * (1 - self.smooth_factor)

        loss = F.binary_cross_entropy_with_logits(label_input, label_target, reduction="none")

        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss
