import numpy as np
import pytest
import torch
from pytorch_toolbelt.inference.functional import unpad_xyxy_bboxes, pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules.encoders import make_n_channel_input
from pytorch_toolbelt.utils import match_bboxes
from torch import nn


def test_unpad_xyxy_bboxes():
    bboxes1 = torch.rand((1, 32, 4))
    bboxes2 = torch.rand((1, 32, 4, 20))

    pad = [2, 3, 4, 5]
    bboxes1_unpad = unpad_xyxy_bboxes(bboxes1, pad, dim=-1)
    assert bboxes1_unpad.size(0) == 1
    assert bboxes1_unpad.size(1) == 32
    assert bboxes1_unpad.size(2) == 4

    bboxes2_unpad = unpad_xyxy_bboxes(bboxes2, pad, dim=2)
    assert bboxes2_unpad.size(0) == 1
    assert bboxes2_unpad.size(1) == 32
    assert bboxes2_unpad.size(2) == 4
    assert bboxes2_unpad.size(3) == 20


def test_make_n_channel_input():
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    conv6 = make_n_channel_input(conv, in_channels=6)
    assert conv6.weight.size(0) == conv.weight.size(0)
    assert conv6.weight.size(1) == 6
    assert conv6.weight.size(2) == conv.weight.size(2)
    assert conv6.weight.size(3) == conv.weight.size(3)

    conv5 = make_n_channel_input(conv, in_channels=5)
    assert conv5.weight.size(0) == conv.weight.size(0)
    assert conv5.weight.size(1) == 5
    assert conv5.weight.size(2) == conv.weight.size(2)
    assert conv5.weight.size(3) == conv.weight.size(3)


@pytest.mark.parametrize(
    ["shape", "padding"],
    [((1, 3, 221, 234), 32), ((1, 3, 256, 256), 32), ((1, 3, 512, 512), 16), ((1, 3, 512, 512), 7)],
)
def test_pad_unpad(shape, padding):
    x = torch.randn(shape)

    x_padded, pad_params = pad_image_tensor(x, pad_size=padding)
    assert x_padded.size(2) % padding == 0
    assert x_padded.size(3) % padding == 0

    y = unpad_image_tensor(x_padded, pad_params)
    assert (x == y).all()


@pytest.mark.parametrize(["shape", "padding"], [((1, 3, 512, 512), (7, 13))])
def test_pad_unpad_nonsymmetric(shape, padding):
    x = torch.randn(shape)

    x_padded, pad_params = pad_image_tensor(x, pad_size=padding)
    assert x_padded.size(2) % padding[0] == 0
    assert x_padded.size(3) % padding[1] == 0

    y = unpad_image_tensor(x_padded, pad_params)
    assert (x == y).all()


@pytest.mark.parametrize(
    (
        "gt_bboxes",
        "pred_bboxes",
        "num_classes",
        "true_positives",
        "false_positives",
        "false_negatives",
        "confusion_matrix",
    ),
    [
        # Perfect match
        (
            ([[10, 10, 20, 20]], [0]),
            ([[10, 10, 20, 20]], [0]),
            1,
            np.array([1]),
            np.array([0]),
            np.array([0]),
            np.array([[1, 0], [0, 0]]),
        ),
        # Class mistmatch
        (
            ([[10, 10, 20, 20]], [0]),
            ([[10, 10, 20, 20]], [1]),
            2,
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        ),
        # Full mismatch
        (
            ([[10, 10, 20, 20]], [0]),
            ([[30, 30, 50, 50]], [0]),
            1,
            np.array([0]),
            np.array([1]),
            np.array([1]),
            np.array([[0, 1], [1, 0]]),
        ),
    ],
)
def test_match_bboxes(
    gt_bboxes, pred_bboxes, num_classes, true_positives, false_positives, false_negatives, confusion_matrix
):
    gt_bboxes, gt_labels = gt_bboxes
    pred_bboxes, pred_labels = pred_bboxes

    gt_bboxes = np.asarray(gt_bboxes)
    gt_labels = np.asarray(gt_labels)

    pred_bboxes = np.asarray(pred_bboxes)
    pred_labels = np.asarray(pred_labels)

    pred_scores = np.ones((len(pred_bboxes)))
    result = match_bboxes(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, num_classes=num_classes, iou_threshold=0.5
    )

    np.testing.assert_equal(result.true_positives, true_positives)
    np.testing.assert_equal(result.false_positives, false_positives)
    np.testing.assert_equal(result.false_negatives, false_negatives)
    np.testing.assert_equal(result.confusion_matrix, confusion_matrix)
