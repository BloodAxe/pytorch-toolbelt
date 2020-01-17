import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smooth_factor=1e-4, ignore_index=None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index

    def forward(self, label_input, label_target):
        not_ignored = label_target != self.ignore_index

        num_classes = label_input.size(1)
        one_hot_target = F.one_hot(label_target.masked_fill(~not_ignored, 0), num_classes).float()
        one_hot_target = one_hot_target * (1 - self.smooth_factor) + (1 - one_hot_target) * self.smooth_factor / (
            num_classes - 1
        )
        log_prb = F.log_softmax(label_input, dim=1)
        loss = -(one_hot_target * log_prb).sum(dim=1)
        return torch.mean(loss * not_ignored.float())
