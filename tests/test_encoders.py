import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules.backbone.inceptionv4 import inceptionv4
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.MobilenetV2Encoder, {}],
        [E.MobilenetV3Encoder, {}],
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
        [E.DenseNet121Encoder, {"pretrained": False}],
        [E.DenseNet161Encoder, {"pretrained": False}],
        [E.DenseNet169Encoder, {"pretrained": False}],
        [E.DenseNet201Encoder, {"pretrained": False}],
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
    for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@torch.no_grad()
@skip_if_no_cuda
def test_inceptionv4_encoder():
    backbone = inceptionv4(pretrained=False)
    backbone.last_linear = None

    net = E.InceptionV4Encoder(pretrained=False, layers=[0, 1, 2, 3, 4]).cuda()

    print(count_parameters(backbone))
    print(count_parameters(net))

    x = torch.randn((4, 3, 512, 512)).cuda()

    out = net(x)
    for fm in out:
        print(fm.size())


@torch.no_grad()
@skip_if_no_cuda
def test_densenet():
    from torchvision.models import densenet121

    net1 = E.DenseNet121Encoder(pretrained=False)
    net2 = densenet121(pretrained=False)
    net2.classifier = None

    print(count_parameters(net1), count_parameters(net2))


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.HRNetV2Encoder18, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.HRNetV2Encoder34, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.HRNetV2Encoder48, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_hrnet_encoder(encoder: E.EncoderModule, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.output_strides)
    print(net.output_filters)
    input = torch.rand((4, 3, 256, 256))
    input = maybe_cuda(input)
    net = maybe_cuda(net)
    output = net(input)
    assert len(output) == len(net.output_filters)
    for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.XResNet18Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.XResNet34Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.XResNet50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.XResNet101Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.XResNet152Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.SEXResNet18Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.SEXResNet34Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.SEXResNet50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.SEXResNet101Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [E.SEXResNet152Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_xresnet_encoder(encoder, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.output_strides)
    print(net.output_filters)
    input = torch.rand((4, 3, 256, 256))
    input = maybe_cuda(input)
    net = maybe_cuda(net)
    output = net(input)
    assert len(output) == len(net.output_filters)
    for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(["encoder", "encoder_params"], [[E.StackedHGEncoder, {}]])
@torch.no_grad()
@skip_if_no_cuda
def test_hourglass_encoder(encoder, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.output_strides)
    print(net.output_filters)
    input = torch.rand((4, 3, 256, 256))
    input = maybe_cuda(input)
    net = maybe_cuda(net)
    output = net(input)
    assert len(output) == len(net.output_filters)
    for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(["encoder", "encoder_params"], [[E.StackedSupervisedHGEncoder, {"supervision_channels": 1}]])
@torch.no_grad()
@skip_if_no_cuda
def test_supervised_hourglass_encoder(encoder, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.output_strides)
    print(net.output_filters)
    input = torch.rand((4, 3, 256, 256))
    input = maybe_cuda(input)
    net = maybe_cuda(net)
    output, supervision = net(input)
    assert len(output) == len(net.output_filters)
    assert len(supervision) == len(net.output_filters) - 2

    for feature_map, expected_stride, expected_channels in zip(output, net.output_strides, net.output_filters):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256

    # for feature_map, expected_stride, expected_channels in zip(supervision, net.output_strides, net.output_filters):
    #     assert feature_map.size(1) == expected_channels
    #     assert feature_map.size(2) * expected_stride == 256
    #     assert feature_map.size(3) * expected_stride == 256
