import numpy as np
import pytest
import pytorch_toolbelt.losses.functional as F
import torch
from pytorch_toolbelt import losses as L
from torch import nn

from pytorch_toolbelt.losses import BinaryFocalLoss, CrossEntropyFocalLoss


def test_sigmoid_focal_loss_with_logits():
    input_good = torch.tensor([10, -10, 10]).float()
    input_bad = torch.tensor([-1, 2, 0]).float()
    target = torch.tensor([1, 0, 1])

    loss_good = F.focal_loss_with_logits(input_good, target)
    loss_bad = F.focal_loss_with_logits(input_bad, target)
    assert loss_good < loss_bad

    loss_cls = BinaryFocalLoss()
    assert loss_cls(input_good, target) < loss_cls(input_bad, target)


def test_softmax_focal_loss_with_logits():
    input_good = torch.tensor([[0, 10, 0], [10, 0, 0], [0, 0, 10]]).float()
    input_bad = torch.tensor([[0, -10, 0], [0, 10, 0], [0, 0, 10]]).float()
    target = torch.tensor([1, 0, 2]).long()

    loss_good = F.softmax_focal_loss_with_logits(input_good, target)
    loss_bad = F.softmax_focal_loss_with_logits(input_bad, target)
    assert loss_good < loss_bad

    loss_cls = CrossEntropyFocalLoss()
    assert loss_cls(input_good, target) < loss_cls(input_bad, target)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 0.5, 1e-5],
    ],
)
def test_soft_jaccard_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 1, 0, 0], [0, 0, 1, 1]], 1.0, 1e-5],
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[0, 0, 1, 0], [0, 1, 0, 0]], 0.0, 1e-5],
        [[[1, 1, 0, 0], [0, 0, 0, 1]], [[1, 1, 0, 0], [0, 0, 0, 0]], 0.5, 1e-5],
    ],
)
def test_soft_jaccard_score_2(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, dims=[1], eps=eps)
    actual = actual.mean()
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 2.0 / 3.0, 1e-5],
    ],
)
def test_soft_dice_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_dice_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@torch.no_grad()
def test_dice_loss_binary():
    eps = 1e-5
    criterion = L.DiceLoss(mode="binary", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)


@torch.no_grad()
def test_binary_jaccard_loss():
    eps = 1e-5
    criterion = L.JaccardLoss(mode="binary", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0]).view(1, 1, 1, 1)
    y_true = torch.tensor(([1])).view(1, 1, 1, 1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)


@torch.no_grad()
def test_multiclass_jaccard_loss():
    eps = 1e-5
    criterion = L.JaccardLoss(mode="multiclass", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[0, 0, 1, 1]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=eps)


@torch.no_grad()
def test_multilabel_jaccard_loss():
    eps = 1e-5
    criterion = L.JaccardLoss(mode="multilabel", from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = 1 - y_pred
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=eps)


@torch.no_grad()
def test_soft_ce_loss():
    soft_ce_criterion = L.SoftCrossEntropyLoss(smooth_factor=0.0)
    ce_criterion = nn.CrossEntropyLoss()

    y_pred = torch.tensor([[+1, -1, -1, -1], [-1, +1, -1, -1], [-1, -1, +1, -1], [-1, -1, -1, +1]]).float()
    y_true = torch.tensor([0, 1, 2, 3]).long()

    actual = soft_ce_criterion(y_pred, y_true).item()
    expected = ce_criterion(y_pred, y_true).item()
    np.testing.assert_almost_equal(actual, expected)


@torch.no_grad()
def test_soft_bce_loss():
    criterion = L.SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=-100)

    # Ideal case
    y_pred = torch.tensor([-9, 9, 1, 9, -9]).float()
    y_true = torch.tensor([0, 1, -100, 1, 0]).long()

    loss = criterion(y_pred, y_true)
    print(loss)


@pytest.mark.parametrize(
    "criterion",
    [
        L.BiTemperedLogisticLoss(t1=1, t2=0.8),
        L.FocalCosineLoss(),
        L.BinaryFocalLoss(),
        L.CrossEntropyFocalLoss(),
        L.SoftF1Loss(),
        L.SoftCrossEntropyLoss(),
    ],
)
def test_classification_losses(criterion):
    # Ideal case
    y_pred = torch.tensor([[+9, -9, -9, -9], [-9, +9, -9, -9], [-9, -9, +9, -9], [-9, -9, -9, +9]]).float()
    y_true = torch.tensor([0, 1, 2, 3]).long()

    loss = criterion(y_pred, y_true)
    print(loss)


def test_binary_bi_tempered_loss():
    loss = L.BinaryBiTemperedLogisticLoss(t1=0.9, t2=3.0, ignore_index=-100)

    y_pred = torch.randn((4, 1, 512, 512))
    y_true = (y_pred > 0).type_as(y_pred)
    y_true[:, :, ::10, ::20] = -100
    loss_value = loss(y_pred, y_true)
    assert len(loss_value.size()) == 0


def test_bbce():
    x = torch.tensor([0, 0, 0, 0, 0]).float()
    y = torch.tensor([0, 1, 1, 1, 1]).float()
    loss = L.balanced_binary_cross_entropy_with_logits(x, y, gamma=1, reduction="none")
    print(loss)
