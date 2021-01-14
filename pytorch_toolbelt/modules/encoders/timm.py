import math
import warnings
from collections import OrderedDict
from typing import List

import torch
from torch import nn

from pytorch_toolbelt.modules import Swish
from .common import EncoderModule, make_n_channel_input, _take

__all__ = [
    "B0Encoder",
    "B1Encoder",
    "B2Encoder",
    "B3Encoder",
    "B4Encoder",
    "B5Encoder",
    "B6Encoder",
    "B7Encoder",
    "MixNetXLEncoder",
    "HRNetW32Encoder",
    "HRNetW18Encoder",
    "HRNetW48Encoder",
    "SKResNet18Encoder",
    "SKResNeXt50Encoder",
    "SWSLResNeXt101Encoder",
    "TResNetMEncoder",
    "DPN68Encoder",
    "DPN68BEncoder",
    "DPN92Encoder",
    "DPN107Encoder",
    "DPN131Encoder",
]


def make_n_channel_input_conv2d_same(conv: nn.Conv2d, in_channels: int, mode="auto", **kwargs):
    assert isinstance(conv, nn.Conv2d)
    if conv.in_channels == in_channels:
        warnings.warn("make_n_channel_input call is spurious")
        return conv

    from timm.models.layers import Conv2dSame

    new_conv = Conv2dSame(
        in_channels,
        out_channels=conv.out_channels,
        kernel_size=kwargs.get("kernel_size", conv.kernel_size),
        stride=kwargs.get("stride", conv.stride),
        padding=kwargs.get("padding", conv.padding),
        dilation=kwargs.get("dilation", conv.dilation),
        groups=kwargs.get("groups", conv.groups),
        bias=kwargs.get("bias", conv.bias is not None),
    )

    w = conv.weight
    if in_channels > conv.in_channels:
        n = math.ceil(in_channels / float(conv.in_channels))
        w = torch.cat([w] * n, dim=1)
        w = w[:, :in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)
    else:
        w = w[:, 0:in_channels, ...]
        new_conv.weight = nn.Parameter(w, requires_grad=True)

    return new_conv


class B0Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b0_ns

        encoder = tf_efficientnet_b0_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.05
        )
        strides = [2, 4, 8, 16, 32]

        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8

        super().__init__([16, 24, 40, 112, 320], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B1Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b1_ns

        encoder = tf_efficientnet_b1_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.05
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([16, 24, 40, 112, 320], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B2Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b2_ns

        encoder = tf_efficientnet_b2_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.1
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([16, 24, 48, 120, 352], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B3Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b3_ns

        encoder = tf_efficientnet_b3_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.1
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([24, 32, 48, 136, 384], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B4Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b4_ns

        encoder = tf_efficientnet_b4_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([24, 32, 56, 160, 448], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B5Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b5_ns

        encoder = tf_efficientnet_b5_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([24, 40, 64, 176, 512], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B6Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b6_ns

        encoder = tf_efficientnet_b6_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([32, 40, 72, 200, 576], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class B7Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b7_ns

        encoder = tf_efficientnet_b7_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        strides = [2, 4, 8, 16, 32]
        if no_stride:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)

            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 8
        super().__init__([32, 48, 80, 224, 640], strides, layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class MixNetXLEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], act_layer=Swish):
        from timm.models.efficientnet import mixnet_xl

        encoder = mixnet_xl(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2)
        super().__init__([40, 48, 64, 192, 320], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        features = self.encoder(x)
        return _take(features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class TResNetMEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models import tresnet_m

        encoder = tresnet_m(pretrained=pretrained)

        super().__init__([64, 64, 128, 1024, 2048], [4, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.body.SpaceToDepth, encoder.body.conv1)

        self.layer1 = encoder.body.layer1
        self.layer2 = encoder.body.layer2
        self.layer3 = encoder.body.layer3
        self.layer4 = encoder.body.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]


class SKResNet18Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, no_first_max_pool=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models import skresnet18

        encoder = skresnet18(pretrained=pretrained, features_only=True)
        super().__init__([64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(
            OrderedDict([("conv1", encoder.conv1), ("bn1", encoder.bn1), ("act1", encoder.act1)])
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2) if no_first_max_pool else encoder.maxpool,
            encoder.layer1,
        )
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class SKResNeXt50Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models import skresnext50_32x4d

        encoder = skresnext50_32x4d(pretrained=pretrained)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(
            OrderedDict([("conv1", encoder.conv1), ("bn1", encoder.bn1), ("act1", encoder.act1)])
        )

        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class SWSLResNeXt101Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models.resnet import swsl_resnext101_32x8d

        encoder = swsl_resnext101_32x8d(pretrained=pretrained)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.act1)

        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class HRNetW18Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import hrnet

        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = hrnet.hrnet_w18(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([128, 256, 512, 1024], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class HRNetW32Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import hrnet

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = hrnet.hrnet_w32(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([128, 256, 512, 1024], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class HRNetW48Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import hrnet

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = hrnet.hrnet_w48(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([128, 256, 512, 1024], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class DPN68Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn68(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([144, 320, 704, 832], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN68BEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn68b(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([144, 320, 704, 832], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN92Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn92(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([336, 704, 1552, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN107Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn107(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([376, 1152, 2432, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN131Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn131(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([352, 832, 1984, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self
