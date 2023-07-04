import torch
from torch import nn, Tensor


class QualityFocalLoss(nn.Module):
    def __init__(self, beta: float = 2, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    @torch.cuda.amp.autocast(False)
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """

        :param predictions: Prediction Logits
        :param targets: Targets
        :return:
        """
        bce = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        focal_term = torch.nn.functional.l1_loss(predictions.sigmoid(), targets, reduction="none").pow_(self.beta)
        loss = focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
