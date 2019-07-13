from typing import List

import torch
from pytorch_toolbelt.utils.torch_utils import to_numpy, to_tensor
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .functional import soft_jaccard_score

__all__ = ['JaccardLoss']

BINARY_MODE = 'binary'
MULTICLASS_MODE = 'multiclass'
MULTILABEL_MODE = 'multilabel'


class JaccardLoss(_Loss):
    """
    Implementation of Jaccard loss for image segmentation task.
    It supports binary, multi-class and multi-label cases.
    """

    def __init__(self,
                 mode: str,
                 classes: List[int] = None,
                 log_loss=False,
                 from_logits=True,
                 smooth=0,
                 eps=1e-7):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(JaccardLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, 'Masking classes is not supported with mode=binary'
            classes = to_tensor(classes, dtype=torch.long)

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
            # Apply activations to get [0..1] class probabilities
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.softmax(dim=1)
            else:
                y_pred = y_pred.sigmoid()

        num_classes = y_pred.size(1)

        dims = None

        if self.mode == BINARY_MODE:
            mask = torch.sum(y_true) > 0

        if self.mode == MULTICLASS_MODE:
            y_true = torch.eye(num_classes)[y_true.squeeze(1)]
            y_true = y_true.permute(0, 3, 1, 2).type(y_pred.dtype)
            dims = (0,) + tuple(range(2, y_true.ndimension()))

        if self.classes is not None:
            y_pred = y_pred[:, self.classes, ...]
            y_true = y_true[:, self.classes, ...]
            dims = (0,) + tuple(range(2, y_true.ndimension()))

        mask = torch.sum(y_true, dims) > 0
        scores = soft_jaccard_score(y_pred, y_true,
                                    self.smooth, self.eps, dims=dims)

        # Since IoU is not defined when no samples are present,
        # we select classes with non-zero pixels counts
        scores = scores[mask]

        if len(scores) == 0:
            # If IoU is not defined, return zero loss
            return 0

        if len(scores) > 1:
            # If scores is vector, compute a mean of non-nan elements
            scores = scores.mean()

        if self.log_loss:
            loss = -torch.log(scores)
        else:
            loss = 1.0 - scores

        return loss
