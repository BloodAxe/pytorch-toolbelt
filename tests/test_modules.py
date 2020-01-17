import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules.backbone.inceptionv4 import inceptionv4
from pytorch_toolbelt.modules.fpn import HFF
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


def test_hff_dynamic_size():
    feature_maps = [
        torch.randn((4, 3, 512, 512)),
        torch.randn((4, 3, 256, 256)),
        torch.randn((4, 3, 128, 128)),
        torch.randn((4, 3, 64, 64)),
    ]

    hff = HFF(upsample_scale=2)
    output = hff(feature_maps)
    assert output.size(2) == 512
    assert output.size(3) == 512


def test_hff_static_size():
    feature_maps = [
        torch.randn((4, 3, 512, 512)),
        torch.randn((4, 3, 384, 384)),
        torch.randn((4, 3, 256, 256)),
        torch.randn((4, 3, 128, 128)),
        torch.randn((4, 3, 32, 32)),
    ]

    hff = HFF(sizes=[(512, 512), (384, 384), (256, 256), (128, 128), (32, 32)])
    output = hff(feature_maps)
    assert output.size(2) == 512
    assert output.size(3) == 512
