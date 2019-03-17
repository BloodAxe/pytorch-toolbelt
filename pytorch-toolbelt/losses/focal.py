import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__(weight=torch.tensor(alpha, dtype=torch.float) if alpha is not None else None)
        self.gamma = gamma
        # self.alpha = alpha
        # if isinstance(alpha, (float, int)):
        #     self.alpha = torch.Tensor([alpha, 1 - alpha], dtype=torch.float)
        # if isinstance(alpha, list):
        #     self.alpha = torch.Tensor(alpha, dtype=torch.float)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.weight is not None:  # self.weight is alpha in paper
            # if self.alpha.type() != input.data.type():
            #     self.alpha = self.alpha.type_as(input.data)
            at = self.weight.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
