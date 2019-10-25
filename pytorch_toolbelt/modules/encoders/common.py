"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""

from typing import List

from torch import nn


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


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
    def encoder_layers(self):
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)
