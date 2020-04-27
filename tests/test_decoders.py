import pytest
import torch
from torch import nn

import pytorch_toolbelt.modules.encoders as E
import pytorch_toolbelt.modules.decoders as D
from pytorch_toolbelt.modules import FPNFuse
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.utils.torch_utils import count_parameters

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


@torch.no_grad()
def test_unet_encoder_decoder():
    encoder = E.UnetEncoder(3, 32, 5)
    decoder = D.UNetDecoder(encoder.channels)
    input = torch.randn((16, 3, 256, 256)).cuda()
    model = nn.Sequential(encoder, decoder).cuda()
    output = model(input)

    print(count_parameters(encoder))
    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_unet_decoder():
    encoder = E.Resnet18Encoder(pretrained=False, layers=[0, 1, 2, 3, 4])
    decoder = D.UNetDecoder(encoder.channels)
    input = torch.randn((16, 3, 256, 256)).cuda()
    model = nn.Sequential(encoder, decoder).cuda()
    output = model(input)

    print(count_parameters(encoder))
    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_sum():
    channels = [256, 512, 1024, 2048]
    sizes = [64, 32, 16, 8]

    decoder = FPNSumDecoder(channels, 5).eval()

    input = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    outputs = decoder(input)

    print(count_parameters(decoder))
    for o in outputs:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_sum_with_encoder():
    input = torch.randn((16, 3, 256, 256)).cuda()
    encoder = E.Resnet18Encoder(pretrained=False)
    decoder = FPNSumDecoder(encoder.channels, 128)
    model = nn.Sequential(encoder, decoder).cuda()

    output = model(input)

    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_cat_with_encoder():
    input = torch.randn((16, 3, 256, 256)).cuda()
    encoder = E.Resnet18Encoder(pretrained=False)
    decoder = FPNCatDecoder(encoder.channels, 128)
    model = nn.Sequential(encoder, decoder).cuda()

    output = model(input)

    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())

    fuse = FPNFuse()
    o = fuse(output)
    print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_cat():
    channels = [256, 512, 1024, 2048]
    sizes = [64, 32, 16, 8]

    net = FPNCatDecoder(channels, 5).eval()

    input = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    outputs = net(input)

    print(count_parameters(net))
    for output in outputs:
        print(output.size(), output.mean(), output.std())
