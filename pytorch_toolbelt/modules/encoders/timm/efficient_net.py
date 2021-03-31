import math
import warnings

import torch
from torch import nn

from ..common import EncoderModule, _take
from ...activations import ACT_SILU, get_activation_block

__all__ = [
    "TimmB0Encoder",
    "TimmB1Encoder",
    "TimmB2Encoder",
    "TimmB3Encoder",
    "TimmB4Encoder",
    "TimmB5Encoder",
    "TimmB6Encoder",
    "TimmB7Encoder",
    "TimmMixNetXLEncoder",
    "B0Encoder",
    "B1Encoder",
    "B2Encoder",
    "B3Encoder",
    "B4Encoder",
    "B5Encoder",
    "B6Encoder",
    "B7Encoder",
    "MixNetXLEncoder",
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


class TimmB0Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b0_ns

        act_layer = get_activation_block(activation)
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

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self


class TimmB1Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b1_ns

        act_layer = get_activation_block(activation)
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


class TimmB2Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b2_ns

        act_layer = get_activation_block(activation)
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


class TimmB3Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b3_ns

        act_layer = get_activation_block(activation)
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


class TimmB4Encoder(EncoderModule):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        no_stride_s32=False,
        no_stride_s16=False,
    ):
        from timm.models.efficientnet import tf_efficientnet_b4_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b4_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.2
        )
        strides = [2, 4, 8, 16, 32]

        if no_stride_s16:
            encoder.blocks[3][0].conv_dw.stride = (1, 1)
            encoder.blocks[3][0].conv_dw.dilation = (2, 2)
            strides[3] = 8
            strides[4] = 16

        if no_stride_s32:
            encoder.blocks[5][0].conv_dw.stride = (1, 1)
            encoder.blocks[5][0].conv_dw.dilation = (2, 2)
            strides[4] = strides[3]

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


class TimmB5Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b5_ns

        act_layer = get_activation_block(activation)
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


class TimmB6Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b6_ns

        act_layer = get_activation_block(activation)
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


class TimmB7Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, no_stride=False):
        from timm.models.efficientnet import tf_efficientnet_b7_ns

        act_layer = get_activation_block(activation)
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


class TimmMixNetXLEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU):
        from timm.models.efficientnet import mixnet_xl

        act_layer = get_activation_block(activation)
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


# Aliases to keep backward compatibility
B0Encoder = TimmB0Encoder
B1Encoder = TimmB1Encoder
B2Encoder = TimmB2Encoder
B3Encoder = TimmB3Encoder
B4Encoder = TimmB4Encoder
B5Encoder = TimmB5Encoder
B6Encoder = TimmB6Encoder
B7Encoder = TimmB7Encoder
MixNetXLEncoder = TimmMixNetXLEncoder
