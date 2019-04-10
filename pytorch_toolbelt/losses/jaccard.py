from typing import List

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .functional import soft_jaccard_score

__all__ = ['BinaryJaccardLoss', 'BinaryJaccardLogLoss', 'MulticlassJaccardLoss']


class BinaryJaccardLoss(_Loss):
    """Implementation of Jaccard loss for binary image segmentation task
    """

    def __init__(self, from_logits=True, weight=None, smooth=1e-3):
        super(BinaryJaccardLoss, self).__init__()
        self.from_logits = from_logits
        self.weight = weight
        self.smooth = smooth

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if y_true.sum() == 0:
            return 0

        iou = soft_jaccard_score(y_pred, y_true, from_logits=self.from_logits, smooth=self.smooth)
        loss = (1.0 - iou)

        return loss


class BinaryJaccardLogLoss(_Loss):
    """Implementation of Jaccard loss for binary image segmentation task
    """

    def __init__(self, from_logits=True, weight=None, smooth=1e-3):
        super(BinaryJaccardLogLoss, self).__init__()
        self.from_logits = from_logits
        self.weight = weight
        self.smooth = smooth

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if y_true.sum() == 0:
            return 0

        iou = soft_jaccard_score(y_pred, y_true, from_logits=self.from_logits, smooth=self.smooth)
        loss = - torch.log(iou)
        return loss


class MulticlassJaccardLoss(_Loss):
    """Implementation of Jaccard loss for multiclass (semantic) image segmentation task
    """

    def __init__(self, classes: List[int] = None, from_logits=True, weight=None, reduction='elementwise_mean'):
        super(MulticlassJaccardLoss, self).__init__(reduction=reduction)
        self.classes = classes
        self.from_logits = from_logits
        self.weight = weight

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if self.from_logits:
            y_pred = y_pred.softmax(dim=1)

        n_classes = y_pred.size(1)
        smooth = 1e-3

        loss = torch.zeros(n_classes, dtype=torch.float, device=y_pred.device)

        if self.classes is None:
            classes = range(n_classes)
        else:
            classes = self.classes

        if self.weight is None:
            weights = [1] * n_classes
        else:
            weights = self.weight

        for class_index, weight in zip(classes, weights):

            jaccard_target = (y_true == class_index).float()
            jaccard_output = y_pred[:, class_index, ...]

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[class_index] = 0
            else:
                iou = soft_jaccard_score(jaccard_output, jaccard_target, from_logits=False, smooth=smooth)
                loss[class_index] = (1.0 - iou) * weight

        if self.reduction == 'elementwise_mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss
