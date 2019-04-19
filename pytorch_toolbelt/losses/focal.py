import torch
from torch import nn

from .functional import sigmoid_focal_loss
from torch.nn.modules.loss import _Loss

__all__ = ['BinaryFocalLoss', 'FocalLoss']


class BinaryFocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = sigmoid_focal_loss(label_input, label_target, gamma=self.gamma, alpha=self.alpha)
        return loss


class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, label_input, label_target):
        """Compute focal loss for multi-class problem.
        Ignores anchors having -1 target label
        """
        num_classes = label_input.size(1)
        loss = 0
        label_target = label_target.view(-1)
        label_input = label_input.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = label_target != self.ignore

        for cls in range(num_classes):
            cls_label_target = (label_target == (cls + 0)).long()
            cls_label_input = label_input[..., cls]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
        return loss

# Needs testing
# class SoftmaxFocalLoss(nn.Module):
#     def __init__(self, gamma=2, eps=1e-7):
#         super(SoftmaxFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#
#     @staticmethod
#     def _one_hot(index, classes):
#         size = index.size() + (classes,)
#         view = index.size() + (1,)
#
#         mask = torch.Tensor(*size).fill_(0)
#         index = index.view(*view)
#         ones = 1.
#
#         if isinstance(index, Variable):
#             ones = Variable(torch.Tensor(index.size()).fill_(1))
#             mask = Variable(mask, volatile=index.volatile)
#
#         return mask.scatter_(1, index, ones)
#
#     def forward(self, input, target):
#         y = one_hot(target, input.size(-1))
#         logit = F.softmax(input, dim=-1)
#         logit = logit.clamp(self.eps, 1. - self.eps)
#
#         loss = -1 * y * torch.log(logit)  # cross entropy
#         loss = loss * (1 - logit) ** self.gamma  # focal loss
#
#         return loss.sum()
