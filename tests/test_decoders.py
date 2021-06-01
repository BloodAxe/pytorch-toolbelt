import pytest
import pytorch_toolbelt.modules.decoders as D
import pytorch_toolbelt.modules.encoders as E
import torch
from pytorch_toolbelt.modules import FPNFuse
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@torch.no_grad()
@pytest.mark.parametrize(
    ("decoder_cls", "decoder_params"),
    [
        (D.FPNSumDecoder, {"fpn_channels": 128}),
        (D.FPNCatDecoder, {"fpn_channels": 128}),
        (D.DeeplabV3PlusDecoder, {"aspp_channels": 256, "out_channels": 256}),
        (D.DeeplabV3Decoder, {"aspp_channels": 256, "out_channels": 256}),
    ],
)
def test_decoders(decoder_cls, decoder_params):
    channels = [64, 128, 256, 512, 1024]
    input = [torch.randn((4, channels[i], 256 // (2 ** i), 384 // (2 ** i))).cuda() for i in range(len(channels))]
    decoder = decoder_cls(channels, **decoder_params).cuda().eval()
    output = decoder(input)

    print(decoder.__class__.__name__)
    print(count_parameters(decoder))
    for o in output:
        print(o.size())


@torch.no_grad()
def test_unet_encoder_decoder():
    encoder = E.UnetEncoder(3, 32, 5)
    decoder = D.UNetDecoder(encoder.channels)
    x = torch.randn((2, 3, 256, 256))
    model = nn.Sequential(encoder, decoder).eval()

    output = model(x)

    print(count_parameters(encoder))
    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_unet_decoder():
    encoder = E.Resnet18Encoder(pretrained=False, layers=[0, 1, 2, 3, 4])
    decoder = D.UNetDecoder(encoder.channels)
    x = torch.randn((16, 3, 256, 256))
    model = nn.Sequential(encoder, decoder)

    output = model(x)

    print(count_parameters(encoder))
    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_sum():
    channels = [256, 512, 1024, 2048]
    sizes = [64, 32, 16, 8]

    decoder = FPNSumDecoder(channels, 5).eval()

    x = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    outputs = decoder(x)

    print(count_parameters(decoder))
    for o in outputs:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_sum_with_encoder():
    x = torch.randn((16, 3, 256, 256))
    encoder = E.Resnet18Encoder(pretrained=False)
    decoder = FPNSumDecoder(encoder.channels, 128)
    model = nn.Sequential(encoder, decoder)

    output = model(x)

    print(count_parameters(decoder))
    for o in output:
        print(o.size(), o.mean(), o.std())


@torch.no_grad()
def test_fpn_cat_with_encoder():
    x = torch.randn((16, 3, 256, 256))
    encoder = E.Resnet18Encoder(pretrained=False)
    decoder = FPNCatDecoder(encoder.channels, 128)
    model = nn.Sequential(encoder, decoder)

    output = model(x)

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

    x = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    outputs = net(x)

    print(count_parameters(net))
    for output in outputs:
        print(output.size(), output.mean(), output.std())
