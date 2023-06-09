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


class TimmBaseEfficientNetEncoder(EncoderModule):
    def __init__(self, encoder, features, layers=[1, 2, 3, 4], first_conv_stride_one: bool = False):
        strides = [2, 4, 8, 16, 32]

        if first_conv_stride_one:
            strides = [1, 2, 4, 8, 16]
            encoder.conv_stem.stride = (1, 1)

        # if no_stride_s16:
        #     encoder.blocks[3][0].conv_dw.stride = (1, 1)
        #     encoder.blocks[3][0].conv_dw.dilation = (2, 2)
        #     strides[3] = strides[2]
        #     strides[4] = strides[3] * 2
        #
        # if no_stride_s32:
        #     encoder.blocks[5][0].conv_dw.stride = (1, 1)
        #     encoder.blocks[5][0].conv_dw.dilation = (2, 2)
        #     strides[4] = strides[3]

        super().__init__(features, strides, layers)
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


class TimmB0Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        use_tf=True,
    ):
        from timm.models.efficientnet import tf_efficientnet_b0_ns, efficientnet_b0

        model_cls = tf_efficientnet_b0_ns if use_tf else efficientnet_b0

        act_layer = get_activation_block(activation)
        encoder = model_cls(pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.05)

        super().__init__(
            encoder, features=[16, 24, 40, 112, 320], layers=layers, first_conv_stride_one=first_conv_stride_one
        )


class TimmB1Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self, pretrained=True, layers=[1, 2, 3, 4], activation: str = ACT_SILU, first_conv_stride_one: bool = False
    ):
        from timm.models.efficientnet import tf_efficientnet_b1_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b1_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=0.05
        )
        super().__init__(encoder, [16, 24, 40, 112, 320], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB2Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate: float = 0.1,
    ):
        from timm.models.efficientnet import tf_efficientnet_b2_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b2_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )
        super().__init__(encoder, [16, 24, 48, 120, 352], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB3Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate=0.1,
    ):
        from timm.models.efficientnet import tf_efficientnet_b3_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b3_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )
        super().__init__(encoder, [24, 32, 48, 136, 384], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB4Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate=0.2,
    ):
        from timm.models.efficientnet import tf_efficientnet_b4_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b4_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )

        super().__init__(encoder, [24, 32, 56, 160, 448], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB5Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate=0.2,
    ):
        from timm.models.efficientnet import tf_efficientnet_b5_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b5_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )
        super().__init__(encoder, [24, 40, 64, 176, 512], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB6Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate=0.2,
    ):
        from timm.models.efficientnet import tf_efficientnet_b6_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b6_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )
        super().__init__(encoder, [32, 40, 72, 200, 576], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmB7Encoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate=0.2,
    ):
        from timm.models.efficientnet import tf_efficientnet_b7_ns

        act_layer = get_activation_block(activation)
        encoder = tf_efficientnet_b7_ns(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )
        super().__init__(encoder, [32, 48, 80, 224, 640], layers=layers, first_conv_stride_one=first_conv_stride_one)


class TimmMixNetXLEncoder(TimmBaseEfficientNetEncoder):
    def __init__(
        self,
        pretrained=True,
        layers=[1, 2, 3, 4],
        activation: str = ACT_SILU,
        first_conv_stride_one: bool = False,
        drop_path_rate=0.2,
    ):
        from timm.models.efficientnet import mixnet_xl

        act_layer = get_activation_block(activation)
        encoder = mixnet_xl(
            pretrained=pretrained, features_only=True, act_layer=act_layer, drop_path_rate=drop_path_rate
        )
        super().__init__(encoder, [40, 48, 64, 192, 320], layers=layers, first_conv_stride_one=first_conv_stride_one)


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
