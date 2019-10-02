import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules.fpn import HFF
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is not available"
)


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.Resnet34Encoder, {"pretrained": False}],
        [E.Resnet50Encoder, {"pretrained": False}],
        [E.SEResNeXt50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.SEResnet50Encoder, {"pretrained": False}],
        [E.Resnet152Encoder, {"pretrained": False}],
        [E.Resnet101Encoder, {"pretrained": False}],
        [E.SEResnet152Encoder, {"pretrained": False}],
        [E.SEResNeXt101Encoder, {"pretrained": False}],
        [E.SEResnet101Encoder, {"pretrained": False}],
        [E.SENet154Encoder, {"pretrained": False}],
        [E.WiderResnet16Encoder, {}],
        [E.WiderResnet20Encoder, {}],
        [E.WiderResnet38Encoder, {}],
        [E.WiderResnet16A2Encoder, {}],
        [E.WiderResnet20A2Encoder, {}],
        [E.WiderResnet38A2Encoder, {}],
        [E.EfficientNetB0Encoder, {}],
        [E.EfficientNetB1Encoder, {}],
        [E.EfficientNetB2Encoder, {}],
        [E.EfficientNetB3Encoder, {}],
        [E.EfficientNetB4Encoder, {}],
        [E.EfficientNetB5Encoder, {}],
        [E.EfficientNetB6Encoder, {}],
        [E.EfficientNetB7Encoder, {}],
        [E.DenseNet121Encoder, {}],
        [E.DenseNet161Encoder, {}],
        [E.DenseNet169Encoder, {}],
        [E.DenseNet201Encoder, {}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_encoders(encoder: E.EncoderModule, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.output_strides)
    print(net.output_filters)
    input = torch.rand((4, 3, 256, 256))
    input = maybe_cuda(input)
    net = maybe_cuda(net)
    output = net(input)
    assert len(output) == len(net.output_filters)
    for feature_map, expected_stride, expected_channels in zip(
        output, net.output_strides, net.output_filters
    ):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


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
