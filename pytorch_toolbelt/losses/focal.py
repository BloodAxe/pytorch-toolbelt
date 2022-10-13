from functools import partial
from typing import Optional
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .functional import focal_loss_with_logits, softmax_focal_loss_with_logits
from pytorch_toolbelt.utils import pytorch_toolbelt_deprecated

__all__ = ["SigmoidFocalLoss", "SoftmaxFocalLoss", "BinaryFocalLoss", "FocalLoss"]


class SigmoidFocalLoss(_Loss):
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
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
            ignore_index=ignore_index,
        )

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss for binary classification problem."""
        loss = self.focal_loss_fn(inputs, targets)
        return loss


class SoftmaxFocalLoss(_Loss):
    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        ignore_index: int = -100,
    ):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.reduced_threshold = reduced_threshold
        self.normalized = normalized
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return softmax_focal_loss_with_logits(
            inputs,
            targets,
            gamma=self.gamma,
            reduction=self.reduction,
            normalized=self.normalized,
            reduced_threshold=self.reduced_threshold,
            ignore_index=self.ignore_index,
        )


@pytorch_toolbelt_deprecated("Class BinaryFocalLoss is deprecated. Please use SigmoidFocalLoss instead.")
def BinaryFocalLoss(*input, **kwargs):
    return SigmoidFocalLoss(*input, **kwargs)


@pytorch_toolbelt_deprecated("Class FocalLoss is deprecated. Please use SoftmaxFocalLoss instead.")
def FocalLoss(*input, **kwargs):
    return SoftmaxFocalLoss(*input, **kwargs)
