from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

__all__ = ["MCCLoss"]


class MCCLoss(_Loss):
    """
    Implementation of Matthews Correlation Coefficient (MCC) loss for image
    segmentation task. It supports binary cases.
    Reference: https://github.com/kakumarabhishek/MCC-Loss
    Paper: https://doi.org/10.1109/ISBI48211.2021.9433782
    """

    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = "batch",
        eps: Optional[float] = 1e-7,
    ):
        """
        Initializes the MCCLoss class.

        :param from_logits: Flag to convert logits to probabilities.
                            Default: False.
                            If True, y_pred is assumed to be logits and will
                            be converted to probabilities using logsigmoid.
        :param reduction: Specifies the reduction to apply to the output:
                          - 'sample': compute loss for each sample in the
                                      batch
                          - 'batch': compute loss for the whole batch
                          Default: 'batch'.
        :param eps: Small epsilon for numerical stability.
        """
        super().__init__()
        assert reduction in {"sample", "batch"}, (
            "reduction must be 'sample' or 'batch'"
        )
        self.eps = eps
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Computes the Matthews Correlation Coefficient (MCC) loss.

        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN)),
        where TP, TN, FP, and FN are elements in the confusion matrix.

        :param y_pred: Predicted logits or probabilities.
                       Shape: (N, 1, H, W) or (N, H, W).
                       If `from_logits=True`, logits are converted to
                       probabilities.
        :param y_true: Ground truth labels.
                       Shape: (N, 1, H, W) or (N, H, W).
                       Values should be 0 or 1.
        :return: Computed MCC loss
        """
        # Input validation.
        assert y_pred.shape == y_true.shape, (
            f"y_pred and y_true must have the same shape, "
            f"but got {y_pred.shape} and {y_true.shape}"
        )

        if not self.from_logits:
            assert torch.all(y_pred >= 0) and torch.all(y_pred <= 1), (
                "y_pred must be in [0, 1] range when from_logits=False"
            )

        # Ensure inputs are 4D: (N, 1, H, W)
        if y_pred.ndim == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.ndim == 3:
            y_true = y_true.unsqueeze(1)

        y_true = y_true.float()
        y_pred = y_pred.float()

        # Obtain the batch size.
        batch_size = y_true.shape[0]

        # Flatten spatial dimensions
        y_true = y_true.view(batch_size, -1)
        y_pred = y_pred.view(batch_size, -1)

        # Convert logits to probabilities if needed.
        if self.from_logits:
            # Use logsigmoid to avoid numerical instability.
            y_pred = F.logsigmoid(y_pred).exp()

        # Compute the terms of the confusion matrix.
        if self.reduction == "sample":
            # Sum over the sample dimension for sample-wise reduction.
            dim = 1
        else:
            # Flatten all dimensions and sum once for batch-wise reduction.
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)
            dim = 0

        tp = (y_pred * y_true).sum(dim=dim)
        tn = ((1 - y_pred) * (1 - y_true)).sum(dim=dim)
        fp = (y_pred * (1 - y_true)).sum(dim=dim)
        fn = ((1 - y_pred) * y_true).sum(dim=dim)

        # Special case: perfect predictions.
        # In this case, FP and FN are zero, so we return 0 since the MCC is 1.
        if (fp == 0).all() and (fn == 0).all():
            return tp.new_tensor(0.0)

        # Compute the numerator and denominator of the MCC expression.
        numerator = tp * tn - fp * fn
        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        # Add epsilon to avoid division by zero.
        denominator = torch.sqrt(denominator) + self.eps

        # Compute the MCC loss.
        mcc = numerator / denominator
        loss = 1 - mcc

        return loss.mean()
