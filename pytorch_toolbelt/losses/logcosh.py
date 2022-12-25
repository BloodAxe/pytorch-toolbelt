import torch
from pytorch_toolbelt.losses.functional import log_cosh_loss
from torch import nn

__all__ = ["LogCoshLoss"]


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
