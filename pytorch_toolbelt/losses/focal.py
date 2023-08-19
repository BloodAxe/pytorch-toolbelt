from functools import partial
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from .functional import focal_loss_with_logits, softmax_focal_loss_with_logits
from pytorch_toolbelt.utils import pytorch_toolbelt_deprecated

__all__ = ["CrossEntropyFocalLoss", "BinaryFocalLoss", "FocalLoss"]


class BinaryFocalLoss(nn.Module):
    __constants__ = ["alpha", "gamma", "reduction", "ignore_index", "normalized", "reduced_threshold", "activation"]

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        activation: str = "sigmoid",
        softmax_dim: Optional[int] = None,
    ):
        """

        :param alpha: Prior probability of having positive value in target.
        :param gamma: Power factor for dampening weight (focal strength).
        :param ignore_index: If not None, targets may contain values to be ignored.
        Target values equal to ignore_index will be ignored from loss computation.
        :param reduced: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        :param activation: Either `sigmoid` or `softmax`. If `softmax` is used, `softmax_dim` must be also specified.

        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold
        self.activation = activation
        self.softmax_dim = softmax_dim

        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
            ignore_index=ignore_index,
            activation=activation,
            softmax_dim=softmax_dim,
        )

        self.get_one_hot_targets = (
            self._one_hot_targets_with_ignore if ignore_index is not None else self._one_hot_targets
        )

    def __repr__(self):
        repr = f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma}, "
        repr += f"ignore_index={self.ignore_index}, reduction={self.reduction}, normalized={self.normalized}, "
        repr += f"reduced_threshold={self.reduced_threshold}, activation={self.activation}, "
        repr += f"softmax_dim={self.softmax_dim})"
        return repr

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss for binary classification problem.
        Args:
            inputs: [B,C,H,W]
            targets: [B,C,H,W] one-hot or [B,H,W] long tensor that will be one-hot encoded (w.r.t to ignore_index)

        Returns:

        """

        if len(targets.shape) + 1 == len(inputs.shape):
            targets = self.get_one_hot_targets(targets, num_classes=inputs.size(1))

        loss = self.focal_loss_fn(inputs, targets)
        return loss

    def _one_hot_targets(self, targets, num_classes):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = torch.moveaxis(targets_one_hot, -1, 1)
        return targets_one_hot

    def _one_hot_targets_with_ignore(self, targets, num_classes):
        ignored_mask = targets.eq(self.ignore_index)
        targets_masked = torch.masked_fill(targets, ignored_mask, 0)
        targets_one_hot = torch.nn.functional.one_hot(targets_masked, num_classes=num_classes)
        targets_one_hot = torch.moveaxis(targets_one_hot, -1, 1)
        targets_one_hot.masked_fill_(ignored_mask.unsqueeze(1), self.ignore_index)
        return targets_one_hot


class CrossEntropyFocalLoss(nn.Module):
    """
    Focal loss for multi-class problem. It uses softmax to compute focal term instead of sigmoid as in
    original paper. This loss expects target labes to have one dimension less (like in nn.CrossEntropyLoss).

    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        ignore_index: int = -100,
    ):
        """

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
        """

        Args:
            inputs: [B,C,H,W] tensor
            targets: [B,H,W] tensor

        Returns:

        """
        return softmax_focal_loss_with_logits(
            inputs,
            targets,
            gamma=self.gamma,
            reduction=self.reduction,
            normalized=self.normalized,
            reduced_threshold=self.reduced_threshold,
            ignore_index=self.ignore_index,
        )


@pytorch_toolbelt_deprecated("Class FocalLoss is deprecated. Please use CrossEntropyFocalLoss instead.")
def FocalLoss(*input, **kwargs):
    return CrossEntropyFocalLoss(*input, **kwargs)
