"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""
import math
import warnings
from typing import List, Union, Tuple, Iterable, Any

import torch
from torch import nn, Tensor

__all__ = ["EncoderModule", "_take", "_take_ints", "_take_tensors", "make_n_channel_input"]

from pytorch_toolbelt.utils.support import pytorch_toolbelt_deprecated
from pytorch_toolbelt.modules.interfaces import (
    HasOutputFeaturesSpecification,
    FeatureMapsSpecification,
    AbstractEncoder,
)


def _take(elements: List[Any], indexes: List[int]) -> List[Any]:
    selected = []
    for i in indexes:
        selected.append(elements[i])
    return selected


def _take_ints(elements: List[int], indexes: List[int]) -> List[int]:
    selected: List[int] = []
    for i in indexes:
        selected.append(elements[i])
    return selected


def _take_tensors(elements: List[Tensor], indexes: List[int]) -> List[Tensor]:
    selected: List[Tensor] = []
    for i in indexes:
        selected.append(elements[i])
    return selected


def make_n_channel_input_conv(
    conv: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], in_channels: int, mode="auto", **kwargs
) -> Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]:
    """
    Create convolution block with same parameters and desired number of channels.

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
        padding_mode=kwargs.get("padding_mode", conv.padding_mode),
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


def make_n_channel_input(conv: nn.Module, in_channels: int, mode="auto", **kwargs) -> nn.Module:
    """
    Create convolution block with same parameters and desired number of channels.

    Args:
        conv: Input nn.Conv2D object to copy settings/weights from
        in_channels: Desired number of input channels
        mode:
        **kwargs: Optional overrides for Conv2D parameters
    """
    if isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return make_n_channel_input_conv(conv, in_channels=in_channels, mode=mode, **kwargs)

    raise ValueError(f"Unsupported class {conv.__class__.__name__}")


class EncoderModule(AbstractEncoder):
    __constants__ = ["_layers", "_output_strides", "_output_filters"]

    def __init__(self, channels: List[int], strides: List[int], layers: List[int]):
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("Number of channels must be equal to number of strides")

        self._layers = list(layers)
        self.output_spec = FeatureMapsSpecification(
            channels=_take_ints(channels, layers), strides=_take_ints(strides, layers)
        )

    def forward(self, x: Tensor) -> List[Tensor]:  # skipcq: PYL-W0221
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            x = output
        # Return only features that were requested
        return _take_tensors(output_features, self._layers)

    @property
    @torch.jit.unused
    def channels(self) -> Tuple[int, ...]:
        return self.get_output_spec().channels

    @property
    @torch.jit.unused
    def strides(self) -> Tuple[int, ...]:
        return self.get_output_spec().strides

    @torch.jit.unused
    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)

    @torch.jit.unused
    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        """
        Change number of channels expected in the input tensor. By default,
        all encoders assume 3-channel image in BCHW notation with C=3.
        This method changes first convolution to have user-defined number of
        channels as input.
        """
        raise NotImplementedError

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        return self.output_spec
