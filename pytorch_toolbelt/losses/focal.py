from functools import partial

import torch
from torch.nn.modules.loss import _Loss

from .functional import focal_loss_with_logits

__all__ = ["BinaryFocalLoss", "FocalLoss"]


class BinaryFocalLoss(_Loss):
    def __init__(
        self, alpha=None, gamma=2, ignore_index=None, reduction="mean", normalized=False, reduced_threshold=None
    ):
        """

        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strenght).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param threshold:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem."""
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

            # This check is to prevent NaNs in the loss, when all targets are equal to ignore_index.
            if label_target.numel() == 0:
                return torch.tensor(0, dtype=label_input.dtype, device=label_input.device)

        loss = self.focal_loss_fn(label_input, label_target)
        return loss


class FocalLoss(_Loss):
    def __init__(
        self, alpha=None, gamma=2, ignore_index=None, reduction="mean", normalized=False, reduced_threshold=None
    ):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.focal_loss_fn(cls_label_input, cls_label_target)
        return loss
