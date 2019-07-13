from typing import List

import torch
from pytorch_toolbelt.utils.torch_utils import to_tensor
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .functional import soft_dice_score

__all__ = ['DiceLoss']


class DiceLoss(_Loss):
    """Implementation of Dice loss for image segmentation task. It supports binary, multiclass and multilabel cases
    """

    def __init__(self,
                 mode: str,
                 classes: List[int] = None,
                 log_loss=False,
                 from_logits=True,
                 smooth=0,
                 eps=1e-7):
        assert mode in {'binary', 'multiclass', 'multilabel'}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            classes = to_tensor(classes, dtype=torch.long)
            assert mode != 'binary', 'Masking classes is not supported in mode=binary'

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if self.from_logits:
            if self.mode == 'multiclass':
                y_pred = y_pred.softmax(dim=1)
            else:
                y_pred = y_pred.sigmoid()

        num_classes = y_pred.size(1)

        if self.mode == 'binary':
            y_pred = torch.cat([1 - y_pred, y_pred], dim=1)
            y_true = torch.eye(2)[y_true.squeeze(1)]
            y_true = y_true.permute(0, 3, 1, 2).type(y_pred.dtype)

        if self.mode == 'multiclass':
            y_true = torch.eye(num_classes)[y_true.squeeze(1)]
            y_true = y_true.permute(0, 3, 1, 2).type(y_pred.dtype)

        if self.classes is not None:
            y_pred = y_pred[:, self.classes, ...]
            y_true = y_true[:, self.classes, ...]

        score = soft_dice_score(y_pred, y_true, self.smooth, self.eps)

        if self.log_loss:
            loss = -torch.log(score)
        else:
            loss = 1.0 - score

        return loss
