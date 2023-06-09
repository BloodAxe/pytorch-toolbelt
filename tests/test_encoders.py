import pytest
import torch

import pytorch_toolbelt.modules.encoders as E
from pytorch_toolbelt.modules import AbstractEncoder
from pytorch_toolbelt.modules.backbone.inceptionv4 import inceptionv4
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters, describe_outputs
from pytorch_toolbelt.modules.encoders import timm

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [[E.MobileNetV3Large, {"layers": [0, 1, 2, 3, 4]}], [E.MobileNetV3Small, {"layers": [0, 1, 2, 3, 4]}]],
)
@torch.no_grad()
@skip_if_no_cuda
def test_mobilenetv3_encoders(encoder: E.EncoderModule, encoder_params):
    net = encoder(**encoder_params).eval().change_input_channels(1)
    print(net.__class__.__name__, count_parameters(net))
    print(net.strides)
    print(net.channels)
    x = torch.rand((4, 1, 384, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 384
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.MobilenetV2Encoder, {}],
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
    print(net.strides)
    print(net.channels)
    x = torch.rand((4, 3, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@torch.no_grad()
@skip_if_no_cuda
def test_unet_encoder():
    net = E.UnetEncoder().eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.strides)
    print(net.channels)
    x = torch.rand((4, 3, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        print(feature_map.size(), feature_map.mean(), feature_map.std())
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@torch.no_grad()
@skip_if_no_cuda
def test_inceptionv4_encoder():
    backbone = inceptionv4(pretrained=False)
    backbone.last_linear = None

    net = E.InceptionV4Encoder(pretrained=False, layers=[0, 1, 2, 3, 4]).cuda()

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
    print(net.strides)
    print(net.channels)
    x = torch.rand((4, 3, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
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
    print(net.strides)
    print(net.channels)
    x = torch.rand((4, 3, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [timm.B0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4], "first_conv_stride_one": True}],
        [timm.B1Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B1Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4], "first_conv_stride_one": True}],
        [timm.B2Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B3Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B4Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B5Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B6Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.B7Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.MixNetXLEncoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SKResNet18Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SKResNeXt50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.SWSLResNeXt101Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.TimmResnet200D, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.HRNetW18Encoder, {"pretrained": False}],
        [timm.HRNetW32Encoder, {"pretrained": False}],
        [timm.HRNetW48Encoder, {"pretrained": False}],
        [timm.DPN68Encoder, {"pretrained": False}],
        [timm.DPN68BEncoder, {"pretrained": False}],
        [timm.DPN92Encoder, {"pretrained": False}],
        [timm.DPN107Encoder, {"pretrained": False}],
        [timm.DPN131Encoder, {"pretrained": False}],
        [timm.NFNetF0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF1Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF2Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF3Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF4Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF5Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF6Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFNetF7Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        #
        [timm.NFRegNetB0Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB1Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB2Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB3Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB4Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.NFRegNetB5Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        # Res2Net
        [timm.TimmRes2Net101Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        [timm.TimmRes2Next50Encoder, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
        # EfficientNet V2
        [timm.TimmEfficientNetV2, {"pretrained": False, "layers": [0, 1, 2, 3, 4]}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_timm_encoders(encoder, encoder_params):
    net = encoder(**encoder_params).eval().change_input_channels(5)

    print(net.__class__.__name__, count_parameters(net))
    print(net.strides)
    print(net.channels)
    x = torch.rand((4, 5, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        assert feature_map.size(0) == x.size(0)
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == x.size(2)
        assert feature_map.size(3) * expected_stride == x.size(3)

    print(describe_outputs(output))


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.StackedHGEncoder, {"repeats": 1}],
        [E.StackedHGEncoder, {"repeats": 3}],
        [E.StackedHGEncoder, {"repeats": 1, "stack_level": 4}],
        [E.StackedHGEncoder, {"repeats": 3, "stack_level": 4}],
        [E.StackedHGEncoder, {"repeats": 1, "stack_level": 4, "features": 128}],
        [E.StackedHGEncoder, {"repeats": 3, "stack_level": 4, "features": 128}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_hourglass_encoder(encoder, encoder_params):
    net = encoder(**encoder_params).eval()
    print(repr(net), count_parameters(net))
    print("Strides ", net.strides)
    print("Channels", net.channels)
    x = torch.rand((4, 3, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)
    assert len(output) == len(net.channels)
    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(["encoder", "encoder_params"], [[E.StackedSupervisedHGEncoder, {"supervision_channels": 1}]])
@torch.no_grad()
@skip_if_no_cuda
def test_supervised_hourglass_encoder(encoder, encoder_params):
    net = encoder(**encoder_params).eval()
    print(net.__class__.__name__, count_parameters(net))
    print(net.get_output_spec())

    x = torch.rand((4, 3, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output, supervision = net(x)
    assert len(output) == len(net.channels)
    assert len(supervision) == len(net.channels) - 2

    for feature_map, expected_stride, expected_channels in zip(output, net.strides, net.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == 256
        assert feature_map.size(3) * expected_stride == 256


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.SwinT, {}],
        [E.SwinS, {}],
        [E.SwinB, {}],
        [E.SwinL, {}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_swin_encoder(encoder, encoder_params):
    net = encoder(**encoder_params).change_input_channels(5).eval()
    output_spec = net.get_output_spec()

    print(net.__class__.__name__, count_parameters(net))
    print(output_spec)

    x = torch.rand((4, 5, 256, 256))
    x = maybe_cuda(x)
    net = maybe_cuda(net)
    output = net(x)

    assert len(output) == len(output_spec.channels)

    for feature_map, expected_stride, expected_channels in zip(output, output_spec.strides, output_spec.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == x.size(2)
        assert feature_map.size(3) * expected_stride == x.size(3)


@pytest.mark.parametrize(
    ["encoder", "encoder_params"],
    [
        [E.MitB0Encoder, {"pretrained": True}],
        [E.MitB1Encoder, {"pretrained": True}],
        [E.MitB2Encoder, {"pretrained": True}],
        [E.MitB3Encoder, {"pretrained": True}],
        [E.MitB4Encoder, {"pretrained": True}],
        [E.MitB5Encoder, {"pretrained": True}],
    ],
)
@torch.no_grad()
@skip_if_no_cuda
def test_mit_encoder(encoder, encoder_params):
    encoder: AbstractEncoder = encoder(**encoder_params).change_input_channels(5).eval()
    print(encoder.__class__.__name__, count_parameters(encoder, human_friendly=True))

    output_spec = encoder.get_output_spec()
    print(encoder.get_output_spec())
    x = torch.rand((4, 5, 256, 256))
    x = maybe_cuda(x)
    encoder = maybe_cuda(encoder)
    output = encoder(x)

    assert len(output) == len(output_spec.channels)

    for feature_map, expected_stride, expected_channels in zip(output, output_spec.strides, output_spec.channels):
        assert feature_map.size(1) == expected_channels
        assert feature_map.size(2) * expected_stride == x.size(2)
        assert feature_map.size(3) * expected_stride == x.size(3)
