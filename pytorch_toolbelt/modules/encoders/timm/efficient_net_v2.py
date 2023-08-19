import copy
import math
import warnings

import torch
from torch import nn

from .common import GenericTimmEncoder
from ..common import make_n_channel_input
from ...activations import ACT_SILU, get_activation_block

__all__ = ["TimmEfficientNetV2"]


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


class TimmEfficientNetV2(GenericTimmEncoder):
    def __init__(
        self,
        model_name: str = "efficientnetv2_rw_s",
        pretrained=True,
        layers=None,
        activation: str = ACT_SILU,
        drop_rate=0.0,
        drop_path_rate=0.0,
        first_conv_stride_one: bool = False,
    ):
        from timm.models.factory import create_model

        act_layer = get_activation_block(activation)
        encoder = create_model(
            model_name=model_name,
            pretrained=pretrained,
            features_only=True,
            act_layer=act_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        if first_conv_stride_one:
            encoder.conv_stem.stride = (1, 1)
            encoder.feature_info = copy.deepcopy(encoder.feature_info)
            for fi in encoder.feature_info:
                fi["reduction"] = fi["reduction"] // 2

        super().__init__(encoder, layers)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        from timm.models.layers import Conv2dSame

        if isinstance(self.encoder.conv_stem, Conv2dSame):
            self.encoder.conv_stem = make_n_channel_input_conv2d_same(
                self.encoder.conv_stem, input_channels, mode, **kwargs
            )
        else:
            self.encoder.conv_stem = make_n_channel_input(self.encoder.conv_stem, input_channels, mode, **kwargs)
        return self
