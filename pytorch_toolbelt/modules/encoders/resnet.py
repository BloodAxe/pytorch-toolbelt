"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""

from collections import OrderedDict

from torch import nn
from torchvision.models import resnet50, resnet34, resnet18, resnet101, resnet152

from .common import EncoderModule, _take, make_n_channel_input

__all__ = [
    "ResnetEncoder",
    "Resnet18Encoder",
    "Resnet34Encoder",
    "Resnet50Encoder",
    "Resnet101Encoder",
    "Resnet152Encoder",
]


class ResnetEncoder(EncoderModule):
    def __init__(self, resnet, filters, strides, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]

        num_layers = max(layers) + 1
        assert num_layers >= 0 and num_layers <= 5

        super().__init__(filters[:num_layers], strides[:num_layers], layers[:num_layers])

        layer0 = nn.Sequential(OrderedDict([("conv0", resnet.conv1), ("bn0", resnet.bn1), ("act0", resnet.relu)]))

        layers = [
            layer0,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ]
        self.maxpool = resnet.maxpool

        self._encoder_layers = nn.ModuleList(layers[:num_layers])

    @property
    def encoder_layers(self):
        return self._encoder_layers

    def forward(self, x):
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)

            if layer == self.encoder_layers[0]:
                # Fist maxpool operator is not a part of layer0 because we want that layer0 output to have stride of 2
                output = self.maxpool(output)
            input = output

        # Return only features that were requested
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self._encoder_layers[0].conv0 = make_n_channel_input(self._encoder_layers[0].conv0, input_channels, mode)
        return self


class Resnet18Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet18(pretrained=pretrained), [64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)


class Resnet34Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet34(pretrained=pretrained), [64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)


class Resnet50Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet50(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class Resnet101Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet101(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class Resnet152Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(resnet152(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
