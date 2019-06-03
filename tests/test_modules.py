import pytest
import torch
import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters


def test_resnet18_encoder():
    encoder = maybe_cuda(E.Resnet18Encoder(layers=[0, 1, 2, 3, 4]).eval())
    input = maybe_cuda(torch.rand((4, 3, 512, 512)))

    feature_maps = encoder(input)
    assert len(feature_maps) == 5

    assert feature_maps[0].size(2) == 256


@pytest.mark.parametrize(['encoder', 'encoder_params'], [
    [E.SqueezenetEncoder, {'layers': [0, 1, 2, 3]}],
    [E.MobilenetV2Encoder, {'layers': [0, 1, 2, 3, 4, 5, 6, 7]}],
    [E.MobilenetV2Encoder, {'layers': [3, 5, 7], 'activation': 'elu'}],
    [E.MobilenetV3Encoder, {'small': False}],
    [E.MobilenetV3Encoder, {'small': True}],
    [E.Resnet18Encoder, {'layers': [0, 1, 2, 3, 4]}],
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
        for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
            assert feature_map.size(1) == expected_channels
            assert feature_map.size(2) * expected_stride == 512
            assert feature_map.size(3) * expected_stride == 512


@pytest.mark.parametrize(['encoder', 'encoder_params'], [
    [E.Resnet34Encoder, {}],
    [E.Resnet50Encoder, {}],
    [E.SEResNeXt50Encoder, {'layers': [0, 1, 2, 3, 4]}],
    [E.SEResnet50Encoder, {}],
    [E.Resnet152Encoder, {}],
    [E.Resnet101Encoder, {}],
    [E.SEResnet152Encoder, {}],
    [E.SEResNeXt101Encoder, {}],
    [E.SEResnet101Encoder, {}],
    [E.SENet154Encoder, {}],
    [E.WiderResnet16Encoder, {}],
    [E.WiderResnet20Encoder, {}],
    [E.WiderResnet38Encoder, {}],
    [E.WiderResnet16A2Encoder, {}],
    [E.WiderResnet20A2Encoder, {}],
    [E.WiderResnet38A2Encoder, {}],
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
        for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
            assert feature_map.size(1) == expected_channels
            assert feature_map.size(2) * expected_stride == 512
            assert feature_map.size(3) * expected_stride == 512
