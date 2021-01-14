"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""
from typing import List

from torch import Tensor

from .common import EncoderModule, _take, make_n_channel_input
from ..backbone.senet import (
    SENet,
    se_resnext50_32x4d,
    se_resnext101_32x4d,
    se_resnet50,
    se_resnet101,
    se_resnet152,
    senet154,
)

__all__ = [
    "SEResnetEncoder",
    "SEResnet50Encoder",
    "SEResnet101Encoder",
    "SEResnet152Encoder",
    "SEResNeXt50Encoder",
    "SEResNeXt101Encoder",
    "SENet154Encoder",
]


class SEResnetEncoder(EncoderModule):
    """
    The only difference from vanilla ResNet is that it has 'layer0' module
    """

    def __init__(self, seresnet: SENet, channels, strides, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__(channels, strides, layers)

        self.maxpool = seresnet.layer0.pool
        del seresnet.layer0.pool

        self.layer0 = seresnet.layer0
        self.layer1 = seresnet.layer1
        self.layer2 = seresnet.layer2
        self.layer3 = seresnet.layer3
        self.layer4 = seresnet.layer4

        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    def output_strides(self):
        return self._output_strides

    @property
    def output_filters(self):
        return self._output_filters

    def forward(self, x: Tensor) -> List[Tensor]:
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)

            if layer == self.layer0:
                # Fist maxpool operator is not a part of layer0
                # because we want that layer0 output to have stride of 2
                output = self.maxpool(output)
            x = output

        # Return only features that were requested
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode, **kwargs)
        return self


class SEResnet50Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnet50(pretrained="imagenet" if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResnet101Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnet101(pretrained="imagenet" if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResnet152Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnet152(pretrained="imagenet" if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SENet154Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        encoder = senet154(pretrained="imagenet" if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResNeXt50Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnext50_32x4d(pretrained="imagenet" if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResNeXt101Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        encoder = se_resnext101_32x4d(pretrained="imagenet" if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
