from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

__all__ = ["BalancedBCEWithLogitsLoss", "balanced_binary_cross_entropy_with_logits"]


def balanced_binary_cross_entropy_with_logits(
    outputs: Tensor, targets: Tensor, reduction: str = "mean", gamma: float = 1.0
) -> Tensor:
    one_minus_beta: Tensor = targets.mean()
    beta = 1.0 - one_minus_beta

    pos_term = beta.pow(gamma) * targets * torch.nn.functional.logsigmoid(outputs)
    neg_term = one_minus_beta.pow(gamma) * (1 - targets) * torch.nn.functional.logsigmoid(-outputs)

    loss = -(pos_term + neg_term)

    if reduction == "mean":
        loss = loss.mean()

    if reduction == "sum":
        loss = loss.sum()

    return loss


class BalancedBCEWithLogitsLoss(nn.Module):
    """
    Balanced binary cross-entropy loss
    """

    __constants__ = ["gamma", "reduction"]

    def __init__(
        self,
        gamma: float = 1.0,
        reduction="mean",
    ):
        """

        Args:
            gamma:
            ignore_index:
            reduction:
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return balanced_binary_cross_entropy_with_logits(output, target, gamma=self.gamma, reduction=self.reduction)
