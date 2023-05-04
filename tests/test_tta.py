from collections import defaultdict

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.utils.torch_utils import to_numpy

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class SumAll(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sum(dim=[1, 2, 3])


def test_d4_image2mask():
    x = torch.rand((4, 3, 224, 224))
    model = NoOp()

    output = tta.d4_image2mask(model, x)
    np.testing.assert_allclose(to_numpy(output), to_numpy(x), atol=1e-6, rtol=1e-6)


def test_flips_image2mask_v2():
    x = torch.rand((4, 3, 224, 224))
    x_a = tta.flips_image_augment(x)
    y = tta.flips_image_deaugment(x_a)

    np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)


def test_d2_image2mask_v2():
    x = torch.rand((4, 3, 224, 224))
    x_a = tta.d2_image_augment(x)
    y = tta.d2_image_deaugment(x_a)

    np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)


def test_d4_image2mask_v2():
    x = torch.rand((4, 3, 224, 224))
    x_a = tta.d4_image_augment(x)
    y = tta.d4_image_deaugment(x_a)

    np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)


def test_fliplr_image2mask():
    x = torch.rand((4, 3, 224, 224))
    model = NoOp()

    output = tta.fliplr_image2mask(model, x)
    np.testing.assert_allclose(to_numpy(output), to_numpy(x), atol=1e-6, rtol=1e-6)


def test_d4_image2label():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]).unsqueeze(0).unsqueeze(0).float()
    model = SumAll()

    output = tta.d4_image2label(model, x)
    expected = int(x.sum())

    assert int(output) == expected


def test_fliplr_image2label():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]).unsqueeze(0).unsqueeze(0).float()
    model = SumAll()

    output = tta.fliplr_image2label(model, x)
    expected = int(x.sum())

    assert int(output) == expected


def test_fivecrop_image2label():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]).unsqueeze(0).unsqueeze(0).float()
    model = SumAll()

    output = tta.fivecrop_image2label(model, x, (2, 2))
    expected = ((1 + 2 + 5 + 6) + (3 + 4 + 7 + 8) + (9 + 0 + 3 + 4) + (1 + 2 + 5 + 6) + (6 + 7 + 0 + 1)) / 5

    assert int(output) == expected


def test_tencrop_image2label():
    x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]).unsqueeze(0).unsqueeze(0).float()
    model = SumAll()

    output = tta.tencrop_image2label(model, x, (2, 2))
    expected = (2 * ((1 + 2 + 5 + 6) + (3 + 4 + 7 + 8) + (9 + 0 + 3 + 4) + (1 + 2 + 5 + 6) + (6 + 7 + 0 + 1))) / 10

    assert int(output) == expected
