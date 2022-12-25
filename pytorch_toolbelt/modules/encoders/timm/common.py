import math
import warnings
import torch

from typing import List, Union
from torch import Tensor, nn

from ..common import EncoderModule, _take_ints, _take_tensors

__all__ = ["GenericTimmEncoder", "make_n_channel_input_std_conv"]


class GenericTimmEncoder(EncoderModule):
    def __init__(self, timm_encoder: Union[nn.Module, str], layers: List[int] = None, pretrained=True, **kwargs):
        strides = []
        channels = []
        default_layers = []
        if isinstance(timm_encoder, str):
            import timm.models.factory

            timm_encoder = timm.models.factory.create_model(
                timm_encoder, features_only=True, pretrained=pretrained, **kwargs
            )

        for i, fi in enumerate(timm_encoder.feature_info):
            strides.append(fi["reduction"])
            channels.append(fi["num_chs"])
            default_layers.append(i)

        if layers is None:
            layers = default_layers

        super().__init__(channels, strides, layers)
        self.encoder = timm_encoder

    def forward(self, x: Tensor) -> List[Tensor]:
        all_feature_maps = self.encoder(x)
        return _take_tensors(all_feature_maps, self._layers)


def make_n_channel_input_std_conv(conv: nn.Module, in_channels: int, mode="auto", **kwargs) -> nn.Module:
    """
    Return the same convolution class but with desired number of channels

    Args:
        conv: Input nn.Conv2D object to copy settings/weights from
        in_channels: Desired number of input channels
        mode:
        **kwargs: Optional overrides for Conv2D parameters
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
