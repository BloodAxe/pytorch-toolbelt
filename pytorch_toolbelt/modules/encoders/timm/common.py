import math
import warnings
from typing import List, Union

import torch
from torch import Tensor, nn

from ..common import EncoderModule, _take

__all__ = ["GenericTimmEncoder", "make_n_channel_input_std_conv"]


class GenericTimmEncoder(EncoderModule):
    def __init__(self, timm_encoder: nn.Module, layers: List[int] = None):
        strides = []
        channels = []
        default_layers = []

        for i, oi in enumerate(timm_encoder.feature_info.out_indices):
            fi = timm_encoder.feature_info.info[i]
            strides.append(fi["reduction"])
            channels.append(fi["num_chs"])
            default_layers.append(i)

        if layers is None:
            layers = default_layers

        super().__init__(channels, strides, layers)
        self.encoder = timm_encoder

    def forward(self, x: Tensor) -> List[Tensor]:
        return _take(self.encoder(x), self._layers)


def make_n_channel_input_std_conv(
    conv: Union["ScaledStdConv2d", "ScaledStdConv2dSame"], in_channels: int, mode="auto", **kwargs
) -> Union["ScaledStdConv2d", "ScaledStdConv2dSame"]:
    """

    Args:
        conv: Input nn.Conv2D object to copy settings/weights from
        in_channels: Desired number of input channels
        mode:
        **kwargs: Optional overrides for Conv2D parameters

    Returns:

    """
    conv_cls = conv.__class__

    if conv.in_channels == in_channels:
        warnings.warn("make_n_channel_input call is spurious")
        return conv

    new_conv = conv_cls(
        in_channels,
        out_channels=conv.out_channels,
        kernel_size=kwargs.get("kernel_size", conv.kernel_size),
        stride=kwargs.get("stride", conv.stride),
        padding=kwargs.get("padding", conv.padding),
        dilation=kwargs.get("dilation", conv.dilation),
        groups=kwargs.get("groups", conv.groups),
        bias=kwargs.get("bias", conv.bias is not None),
        # gamma=kwargs.get("gamma", conv.gamma),
        eps=kwargs.get("eps", conv.eps),
        use_layernorm=kwargs.get("use_layernorm", conv.use_layernorm),
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
