import torch
import torch.nn.functional as F
from torch import nn


def binary_focal_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      gamma=2.0,
                      alpha=0.25, with_logits=True) -> torch.Tensor:
    """
        Slightly edited version from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py

    :param pred:
    :param target:
    :param gamma:
    :param alpha:
    :param with_logits: True, if pred Tensor is raw logits (-inf,+inf); False if sigmoid(predictions) has been already applied.
    :return:
    """
    if with_logits:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = pred

    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target))
    weight = weight * pt.pow(gamma)
    if with_logits:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
    else:
        loss = F.binary_cross_entropy(pred, target, reduction='none') * weight
    return loss.sum()


def soft_jaccard_score(pred: torch.Tensor, target: torch.Tensor, smooth=1e-3, from_logits=False) -> torch.Tensor:
    if from_logits:
        pred = pred.sigmoid()

    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    iou = intersection / (union - intersection + smooth)
    return iou


def soft_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth=1e-3, from_logits=False) -> torch.Tensor:
    if from_logits:
        pred = pred.sigmoid()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) + smooth
    return 2 * intersection / union
