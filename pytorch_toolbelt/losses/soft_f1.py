import torch
from torch import nn, Tensor
from typing import Optional

__all__ = ["soft_micro_f1", "BinarySoftF1Loss", "SoftF1Loss"]


def soft_micro_f1(preds: Tensor, targets: Tensor, eps=1e-6) -> Tensor:
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.

    Args:
        targets (Tensor): targets array of shape (Num Samples, Num Classes)
        preds (Tensor): probability matrix of shape (Num Samples, Num Classes)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch

    References:
        https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    """

    tp = torch.sum(preds * targets, dim=0)
    fp = torch.sum(preds * (1 - targets), dim=0)
    fn = torch.sum((1 - preds) * targets, dim=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + eps)
    loss = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    return loss.mean()


# TODO: Test
# def macro_double_soft_f1(y, y_hat):
#     """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
#     Use probability values instead of binary predictions.
#     This version uses the computation of soft-F1 for both positive and negative class for each label.
#
#     Args:
#         y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
#
#     Returns:
#         cost (scalar Tensor): value of the cost function for the batch
#     """
#     tp = tf.reduce_sum(y_hat * y, axis=0)
#     fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
#     fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
#     tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
#     soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
#     soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
#     cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
#     cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
#     cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
#     macro_cost = tf.reduce_mean(cost)  # average on all labels
#     return macro_cost


class BinarySoftF1Loss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = None, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        targets = targets.view(-1)
        preds = preds.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]

            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)

        preds = preds.sigmoid().clamp(self.eps, 1 - self.eps)
        return soft_micro_f1(preds.view(-1, 1), targets.view(-1, 1))


class SoftF1Loss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = None, eps=1e-6):
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        preds = preds.softmax(dim=1).clamp(self.eps, 1 - self.eps)
        targets = torch.nn.functional.one_hot(targets, preds.size(1))

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore_index
            preds = preds[not_ignored]
            targets = targets[not_ignored]

            if targets.numel() == 0:
                return torch.tensor(0, dtype=preds.dtype, device=preds.device)

        return soft_micro_f1(preds, targets)
