from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

__all__ = ["MCCLoss"]


class MCCLoss(_Loss):
    """
    Implementation of Matthews Correlation Coefficient (MCC) loss for image segmentation task.
    It supports binary cases.
    Reference: https://github.com/kakumarabhishek/MCC-Loss
    Paper: https://doi.org/10.1109/ISBI48211.2021.9433782
    """

    def __init__(self, eps: Optional[float] = 1e-7):
        """
        Initializes the MCCLoss class.

        :param eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the Matthews Correlation Coefficient (MCC) loss.
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.

        :param y_pred: Predicted probabilities (logits) of shape (N, 1, H, W)
        :param y_true: Ground truth labels of shape (N, 1, H, W)
        :return: Computed MCC loss
        """

        batch_size = y_true.shape[0]

        y_true = y_true.view(batch_size, 1, -1)
        y_pred = y_pred.view(batch_size, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1 - mcc

        return loss