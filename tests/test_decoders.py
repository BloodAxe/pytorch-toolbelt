import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules.backbone.inceptionv4 import inceptionv4
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is not available"
)

@torch.no_grad()
def test_fpn_sum():
    channels = [256, 512, 1024, 2048]
    sizes = [64, 32, 16, 8]

    net = FPNSumDecoder(channels, 5).eval()

    input = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    output, dsv_masks = net(input)

    print(output.size(), output.mean(), output.std())
    for dsv in dsv_masks:
        print(dsv.size(), dsv.mean(), dsv.std())
    print(count_parameters(net))


@torch.no_grad()
def test_fpn_cat():
    channels = [256, 512, 1024, 2048]
    sizes = [64, 32, 16, 8]

    net = FPNCatDecoder(channels, 5).eval()

    input = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    output, dsv_masks = net(input)

    print(output.size(), output.mean(), output.std())
    for dsv in dsv_masks:
        print(dsv.size(), dsv.mean(), dsv.std())
    print(count_parameters(net))
