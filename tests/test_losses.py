import pytest
import torch

import pytorch_toolbelt.losses.functional as F


def test_sigmoid_focal_loss():
    input_good = torch.Tensor([10, -10, 10]).float()
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])

    loss_good = F.sigmoid_focal_loss(input_good, target)
    loss_bad = F.sigmoid_focal_loss(input_bad, target)
    assert loss_good < loss_bad


def test_soft_jaccard_score():
    input_good = torch.Tensor([1, 0, 1]).float()
    input_bad = torch.Tensor([0, 0, 0]).float()
    target = torch.Tensor([1, 0, 1])
    eps = 1e-5

    jaccard_good = F.soft_jaccard_score(input_good, target, smooth=eps)
    assert float(jaccard_good) == pytest.approx(1.0, eps)

    jaccard_bad = F.soft_jaccard_score(input_bad, target, smooth=eps)
    assert float(jaccard_bad) == pytest.approx(0.0, eps)


def test_soft_dice_score():
    input_good = torch.Tensor([1, 0, 1]).float()
    input_bad = torch.Tensor([0, 0, 0]).float()
    target = torch.Tensor([1, 0, 1])
    eps = 1e-5

    dice_good = F.soft_dice_score(input_good, target, smooth=eps)
    assert float(dice_good) == pytest.approx(1.0, eps)

    dice_bad = F.soft_dice_score(input_bad, target, smooth=eps)
    assert float(dice_bad) == pytest.approx(0.0, eps)
