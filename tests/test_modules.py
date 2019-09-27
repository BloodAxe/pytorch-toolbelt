import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules.fpn import HFF
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters


def test_resnet18_encoder():
    encoder = maybe_cuda(E.Resnet18Encoder(layers=[0, 1, 2, 3, 4]).eval())
    input = maybe_cuda(torch.rand((4, 3, 512, 512)))

    feature_maps = encoder(input)
    assert len(feature_maps) == 5

    assert feature_maps[0].size(2) == 256


@pytest.mark.parametrize(['encoder', 'encoder_params'], [
    [E.SqueezenetEncoder, {'pretrained': False, 'layers': [0, 1, 2, 3]}],
    [E.MobilenetV2Encoder, {'layers': [0, 1, 2, 3, 4, 5, 6, 7]}],
    [E.MobilenetV2Encoder, {'layers': [3, 5, 7], 'activation': 'elu'}],
    [E.MobilenetV3Encoder, {'small': False}],
    [E.MobilenetV3Encoder, {'small': True}],
    [E.Resnet18Encoder, {'pretrained': False, 'layers': [0, 1, 2, 3, 4]}],
    [E.EfficientNetB0Encoder, {}],
    [E.EfficientNetB1Encoder, {}]
])
def test_encoders(encoder: E.EncoderModule, encoder_params):
    with torch.no_grad():
        net = encoder(**encoder_params).eval()
        print(net.__class__.__name__, count_parameters(net))
        print(net.output_strides)
        print(net.output_filters)
        input = torch.rand((4, 3, 512, 512))
        input = maybe_cuda(input)
        net = maybe_cuda(net)
        output = net(input)
        assert len(output) == len(net.output_filters)
        for feature_map, expected_stride, expected_channels in zip(output,
                                                                   net.output_strides,
                                                                   net.output_filters):
            assert feature_map.size(1) == expected_channels
            assert feature_map.size(2) * expected_stride == 512
            assert feature_map.size(3) * expected_stride == 512


@pytest.mark.parametrize(['encoder', 'encoder_params'], [
    [E.Resnet34Encoder, {'pretrained': False}],
    [E.Resnet50Encoder, {'pretrained': False}],
    [E.SEResNeXt50Encoder, {'pretrained': False, 'layers': [0, 1, 2, 3, 4]}],
    [E.SEResnet50Encoder, {'pretrained': False}],
    [E.Resnet152Encoder, {'pretrained': False}],
    [E.Resnet101Encoder, {'pretrained': False}],
    [E.SEResnet152Encoder, {'pretrained': False}],
    [E.SEResNeXt101Encoder, {'pretrained': False}],
    [E.SEResnet101Encoder, {'pretrained': False}],
    [E.SENet154Encoder, {'pretrained': False}],
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
    [E.EfficientNetB7Encoder, {}]
])
def test_encoders_cuda_only(encoder: E.EncoderModule, encoder_params):
    if not torch.cuda.is_available():
        return

    with torch.no_grad():
        net = encoder(**encoder_params).eval()
        print(net.__class__.__name__, count_parameters(net))
        print(net.output_strides)
        print(net.output_filters)
        input = torch.rand((4, 3, 512, 512))
        input = maybe_cuda(input)
        net = maybe_cuda(net)
        output = net(input)
        assert len(output) == len(net.output_filters)
        for feature_map, expected_stride, expected_channels in zip(output,
                                                                   net.output_strides,
                                                                   net.output_filters):
            assert feature_map.size(1) == expected_channels
            assert feature_map.size(2) * expected_stride == 512
            assert feature_map.size(3) * expected_stride == 512


def test_densenet_encoder():
    dn121 = E.DenseNet121Encoder(layers=[0, 1, 2, 3, 4])
    out121 = dn121(torch.randn(2, 3, 512, 512))
    print([o.size() for o in out121])

    dn169 = E.DenseNet169Encoder(layers=[0, 1, 2, 3, 4])
    out169 = dn169(torch.randn(2, 3, 512, 512))
    print([o.size() for o in out169])

    dn201 = E.DenseNet201Encoder(layers=[0, 1, 2, 3, 4])
    out201 = dn201(torch.randn(2, 3, 512, 512))
    print([o.size() for o in out201])


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
