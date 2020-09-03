import cv2
import torch
import numpy as np
from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn


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


def test_d4_image2mask_v2():
    x = torch.rand((4, 3, 224, 224))
    x_a = tta.d4_augment(x)
    y = tta.d4_deaugment(x_a)

    np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_d4_speed():
    model = NoOp().eval()

    speed_v1 = 0
    speed_v2 = 0
    for i in range(100):
        x = torch.rand((4, 3, 224, 224)).cuda()
        start = cv2.getTickCount()
        y = tta.d4_image2mask(model, x)
        v = y.sum().item()
        finish = cv2.getTickCount()
        speed_v1 += (finish - start)
        np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)

    for i in range(100):
        x = torch.rand((4, 3, 224, 224)).cuda()
        start = cv2.getTickCount()
        x_a = tta.d4_augment(x)
        x_a = model(x_a)
        y = tta.d4_deaugment(x_a)
        finish = cv2.getTickCount()
        speed_v2 += (finish - start)
        np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)

    print(speed_v1, speed_v2)


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
