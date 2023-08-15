import torch
from torch import nn, Tensor


class QualityFocalLoss(nn.Module):
    __constants__ = ["beta", "reduction"]

    def __init__(self, beta: float = 2, reduction="mean"):
        """
        Quality Focal Loss from https://arxiv.org/abs/2006.04388

        :param beta: Power factor for focal term
        :param reduction: Possible values are: mean, sum, normalized
            mean - mean loss value is returned
            sum - sum of loss values is returned
            normalized - mean loss value is divided by sum of focal terms.
            This mimics normalized focal loss approach.
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    @torch.cuda.amp.autocast(False)
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute quality focal loss

        :param predictions: Prediction logits
        :param targets: Targets of the same shape as predictions
        :return: Loss value
        """
        predictions = predictions.float()
        targets = targets.float()

        bce = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
        focal_term = torch.nn.functional.l1_loss(predictions.sigmoid(), targets, reduction="none").pow_(self.beta)
        loss = focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "normalized":
            return loss.sum() / focal_term.sum()

        return loss
