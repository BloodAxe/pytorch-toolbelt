"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""
import math
from typing import List

import torch
from torch import nn

import warnings

__all__ = ["EncoderModule", "_take", "make_n_channel_input"]


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


def make_n_channel_input(conv: nn.Conv2d, in_channels: int, mode="auto"):
    if conv.in_channels == in_channels:
        warnings.warn("make_n_channel_input call is spurious")
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
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


class EncoderModule(nn.Module):
    def __init__(self, channels: List[int], strides: List[int], layers: List[int]):
        super().__init__()
        assert len(channels) == len(strides)

        self._layers = layers

        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    def forward(self, x):
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)
            input = output
        # Return only features that were requested
        return _take(output_features, self._layers)

    @property
    def output_strides(self) -> List[int]:
        return self._output_strides

    @property
    def output_filters(self) -> List[int]:
        return self._output_filters

    @property
    def encoder_layers(self) -> List[nn.Module]:
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)

    def change_input_channels(self, input_channels: int, mode="auto"):
        """
        Change number of channels expected in the input tensor. By default,
        all encoders assume 3-channel image in BCHW notation with C=3.
        This method changes first convolution to have user-defined number of
        channels as input.
        """
        raise NotImplementedError
