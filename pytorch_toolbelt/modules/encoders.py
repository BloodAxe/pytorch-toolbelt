"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""

from collections import OrderedDict
from typing import List

from torch import nn
from torchvision.models import resnet50, resnet34, resnet18, resnet101, resnet152, squeezenet1_1

from .backbone.mobilenet import MobileNetV2
from .backbone.senet import SENet, se_resnext50_32x4d, se_resnext101_32x4d, se_resnet50, se_resnet101, se_resnet152, senet154


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


class EncoderModule(nn.Module):
    def __init__(self, channels, strides, layers: List[int]):
        super().__init__()
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


class ResnetEncoder(EncoderModule):
    def __init__(self, resnet, filters, strides, layers=[1, 2, 3, 4]):
        super().__init__(filters, strides, layers)

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu)
        ]))
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x):
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)

            if layer == self.layer0:
                # Fist maxpool operator is not a part of layer0 because we want that layer0 output to have stride of 2
                output = self.maxpool(output)
            input = output

        # Return only features that were requested
        return _take(output_features, self._layers)


class Resnet18Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        super().__init__(resnet18(pretrained=pretrained), [64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)


class Resnet34Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        super().__init__(resnet34(pretrained=pretrained), [64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)


class Resnet50Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        super().__init__(resnet50(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class Resnet101Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        super().__init__(resnet101(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class Resnet152Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        super().__init__(resnet152(pretrained=pretrained), [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResnetEncoder(EncoderModule):
    """
    The only difference from vanilla ResNet is that it has 'layer0' module
    """

    def __init__(self, seresnet: SENet, channels, strides, layers=[1, 2, 3, 4]):
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

    def forward(self, x):
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)

            if layer == self.layer0:
                # Fist maxpool operator is not a part of layer0 because we want that layer0 output to have stride of 2
                output = self.maxpool(output)
            input = output

        # Return only features that were requested
        return _take(output_features, self._layers)


class SEResnet50Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = se_resnet50(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResnet101Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = se_resnet101(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResnet152Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = se_resnet152(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SENet154Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = senet154(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResNeXt50Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SEResNeXt101Encoder(SEResnetEncoder):
    def __init__(self, pretrained=True, layers=[1, 2, 3, 4]):
        encoder = se_resnext101_32x4d(pretrained='imagenet' if pretrained else None)
        super().__init__(encoder, [64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)


class SqueezenetEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3]):
        super().__init__([64, 128, 256, 512], [4, 8, 16, 16], layers)
        squeezenet = squeezenet1_1(pretrained=pretrained)

        # nn.Conv2d(3, 64, kernel_size=3, stride=2),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.layer0 = nn.Sequential(
            squeezenet.features[0],
            squeezenet.features[1],
            # squeezenet.features[2],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Fire(64, 16, 64, 64),
        # Fire(128, 16, 64, 64),
        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.layer1 = nn.Sequential(
            squeezenet.features[3],
            squeezenet.features[4],
            # squeezenet.features[5],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Fire(128, 32, 128, 128),
        # Fire(256, 32, 128, 128),
        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.layer2 = nn.Sequential(
            squeezenet.features[6],
            squeezenet.features[7],
            # squeezenet.features[8],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Fire(256, 48, 192, 192),
        # Fire(384, 48, 192, 192),
        # Fire(384, 64, 256, 256),
        # Fire(512, 64, 256, 256),
        self.layer3 = nn.Sequential(
            squeezenet.features[9],
            squeezenet.features[10],
            squeezenet.features[11],
            squeezenet.features[12],
        )

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3]


class MobilenetV2Encoder(EncoderModule):
    def __init__(self, layers=[2, 3, 5, 7]):
        super().__init__([32, 16, 24, 32, 64, 96, 160, 320], [2, 2, 4, 8, 16, 16, 32, 32], layers)
        encoder = MobileNetV2()

        self.layer0 = encoder.layer0
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.layer5 = encoder.layer5
        self.layer6 = encoder.layer6
        self.layer7 = encoder.layer7

    @property
    def encoder_layers(self):
        return [self.layer0,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.layer5,
                self.layer6,
                self.layer7]
