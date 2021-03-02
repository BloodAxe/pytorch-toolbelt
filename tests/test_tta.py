from collections import defaultdict

import cv2
import torch
import numpy as np
import pytest

from torch import nn

from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.modules import GlobalAvgPool2d
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.zoo import resnet34_unet32

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


def test_d4_image2mask_v2():
    x = torch.rand((4, 3, 224, 224))
    x_a = tta.d4_image_augment(x)
    y = tta.d4_image_deaugment(x_a)

    np.testing.assert_allclose(to_numpy(y), to_numpy(x), atol=1e-6, rtol=1e-6)


@torch.no_grad()
@skip_if_no_cuda()
def test_d4_speed():
    df = defaultdict(list)
    n = 100

    model = resnet34_unet32().cuda().eval()
    x = torch.rand((4, 3, 224, 224)).float().cuda()
    y1 = tta.d4_image2mask(model, x)
    y2 = tta.d4_image_deaugment(model(tta.d4_image_augment(x)))
    np.testing.assert_allclose(to_numpy(y1), to_numpy(y2), atol=1e-6, rtol=1e-6)

    for deterministic in [False, True]:
        for benchmark in [False, True]:
            for dtype in [torch.float16, torch.float32]:
                torch.cuda.empty_cache()
                torch.backends.cuda.deterministic = deterministic
                torch.backends.cuda.benchmark = benchmark

                model = resnet34_unet32().to(dtype).cuda().eval()

                speed_v1 = 0
                speed_v2 = 0
                for i in range(n):
                    x = torch.rand((4, 3, 224, 224)).to(dtype).cuda(non_blocking=False)
                    start = cv2.getTickCount()
                    y = tta.d4_image2mask(model, x)
                    v = y.sum().item()
                    finish = cv2.getTickCount()
                    speed_v1 += finish - start
                    np.testing.assert_allclose(v, v, atol=1e-6, rtol=1e-6)

                for i in range(n):
                    x = torch.rand((4, 3, 224, 224)).to(dtype).cuda(non_blocking=False)
                    start = cv2.getTickCount()
                    x_a = tta.d4_image_augment(x)
                    x_a = model(x_a)
                    y = tta.d4_image_deaugment(x_a)
                    v = y.sum().item()
                    finish = cv2.getTickCount()
                    speed_v2 += finish - start
                    np.testing.assert_allclose(v, v, atol=1e-6, rtol=1e-6)

                df["mode"].append("fp16" if dtype == torch.float16 else "fp32")
                df["deterministic"].append(deterministic)
                df["benchmark"].append(benchmark)
                df["d4_image2mask (ms)"].append(1000.0 * speed_v1 / (cv2.getTickFrequency() * n))
                df["d4_augment (ms)"].append(1000.0 * speed_v2 / (cv2.getTickFrequency() * n))

    import pandas as pd

    df = pd.DataFrame.from_dict(df)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(df)
    df.to_csv("tta_eval.csv", index=False)


def test_fliplr_image2label():
    x = torch.rand((4, 3, 224, 224))
    model = GlobalAvgPool2d(flatten=True)

    output = tta.fliplr_image2label(model, x)
    np.testing.assert_allclose(to_numpy(output), to_numpy(x.mean(dim=(2, 3))), atol=1e-6, rtol=1e-6)


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
