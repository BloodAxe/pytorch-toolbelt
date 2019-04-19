import math
import torch
import torch.nn.functional as F

__all__ = ['sigmoid_focal_loss', 'soft_jaccard_score', 'soft_dice_score', 'wing_loss']


def sigmoid_focal_loss(input: torch.Tensor,
                       target: torch.Tensor,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    """Compute binary focal loss between target and output logits.

    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'

    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss


def soft_jaccard_score(pred: torch.Tensor,
                       target: torch.Tensor,
                       smooth=1e-3,
                       from_logits=False) -> torch.Tensor:
    if from_logits:
        pred = pred.sigmoid()

    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    iou = intersection / (union - intersection + smooth)
    return iou


def soft_dice_score(pred: torch.Tensor,
                    target: torch.Tensor,
                    smooth=1e-3,
                    from_logits=False) -> torch.Tensor:
    if from_logits:
        pred = pred.sigmoid()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) + smooth
    return 2 * intersection / union


def wing_loss(prediction: torch.Tensor, target: torch.Tensor, width=5, curvature=0.5, reduction='mean'):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param prediction:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - prediction).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == 'sum':
        loss = loss.sum()

    if reduction == 'mean':
        loss = loss.mean()

    return loss
