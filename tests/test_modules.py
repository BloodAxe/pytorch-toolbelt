import pytest
import torch
from pytorch_toolbelt.modules import SqueezenetEncoder, MobilenetV2Encoder, \
    Resnet152Encoder, Resnet101Encoder, Resnet50Encoder, Resnet34Encoder, Resnet18Encoder, \
    SENet154Encoder, SEResnet152Encoder, SEResnet50Encoder, \
    SEResNeXt101Encoder, SEResNeXt50Encoder, SEResnet101Encoder, EncoderModule
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters


@pytest.mark.parametrize(['encoder', 'encoder_params'], [
    [SqueezenetEncoder, {'layers': [0, 1, 2, 3]}],
    [MobilenetV2Encoder, {'layers': [0, 1, 2, 3, 4, 5, 6, 7]}],
    [Resnet18Encoder, {}],
    [Resnet34Encoder, {}],
    [Resnet50Encoder, {}],
    [Resnet101Encoder, {}],
    [Resnet152Encoder, {}],
    [SEResNeXt50Encoder, {}],
    [SEResNeXt101Encoder, {}],
    [SEResnet50Encoder, {}],
    [SEResnet101Encoder, {}],
    [SEResnet152Encoder, {}],
    [SENet154Encoder, {}],
])
def test_encoders(encoder: EncoderModule, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    with torch.no_grad():
        input = torch.rand((4, 3, 512, 512))
        input = maybe_cuda(input)
        net = maybe_cuda(net)
        output = net(input)
        assert len(output) == len(net.output_filters)
        for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
            assert feature_map.size(1) == expected_channels
            assert feature_map.size(2) * expected_stride == 512
            assert feature_map.size(3) * expected_stride == 512
