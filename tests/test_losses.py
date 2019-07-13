import pytest
import torch

import pytorch_toolbelt.losses.functional as F
from pytorch_toolbelt.losses import DiceLoss, JaccardLoss


def test_sigmoid_focal_loss():
    input_good = torch.Tensor([10, -10, 10]).float()
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])

    loss_good = F.sigmoid_focal_loss(input_good, target)
    loss_bad = F.sigmoid_focal_loss(input_bad, target)
    assert loss_good < loss_bad


def test_reduced_focal_loss():
    input_good = torch.Tensor([10, -10, 10]).float()
    input_bad = torch.Tensor([-1, 2, 0]).float()
    target = torch.Tensor([1, 0, 1])

    loss_good = F.reduced_focal_loss(input_good, target)
    loss_bad = F.reduced_focal_loss(input_bad, target)
    assert loss_good < loss_bad


@pytest.mark.parametrize(['y_true', 'y_pred', 'expected', 'eps'], [
    [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
    [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
    [[1, 1, 1, 1], [1, 1, 0, 0], 0.5, 1e-5],
])
def test_soft_jaccard_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(['y_true', 'y_pred', 'expected', 'eps'], [
    [[[1, 1, 0, 0], [0, 0, 1, 1]],
     [[1, 1, 0, 0], [0, 0, 1, 1]], 1.0, 1e-5],
    [[[1, 1, 0, 0], [0, 0, 1, 1]],
     [[0, 0, 1, 0], [0, 1, 0, 0]], 0.0, 1e-5],
    [[[1, 1, 0, 0], [0, 0, 0, 1]],
     [[1, 1, 0, 0], [0, 0, 0, 0]], 0.5, 1e-5],
])
def test_soft_jaccard_score_2(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, dims=[1], eps=eps)
    actual = actual.mean()
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(['y_true', 'y_pred', 'expected', 'eps'], [
    [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
    [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
    [[1, 1, 1, 1], [1, 1, 0, 0], 0.5, 1e-5],
])
def test_soft_dice_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_dice_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@torch.no_grad()
def test_dice_loss_binary():
    eps = 1e-5
    criterion = DiceLoss(mode='binary', from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)


@torch.no_grad()
def test_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode='binary', from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)
