"""Wrappers for different backbones for models that follows Encoder-Decoder architecture.

Encodes listed here provides easy way to swap backbone of classification/segmentation/detection model.
"""

from collections import OrderedDict
from typing import List

from torch import nn
from torchvision.models import (
    resnet50,
    resnet34,
    resnet18,
    resnet101,
    resnet152,
    squeezenet1_1,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    DenseNet,
)

from pytorch_toolbelt.modules.abn import ABN
from pytorch_toolbelt.modules.backbone.efficient_net import (
    efficient_net_b0,
    efficient_net_b6,
    efficient_net_b1,
    efficient_net_b2,
    efficient_net_b3,
    efficient_net_b4,
    efficient_net_b5,
    efficient_net_b7,
)
from pytorch_toolbelt.modules.backbone.mobilenetv3 import MobileNetV3
from pytorch_toolbelt.modules.backbone.wider_resnet import WiderResNet, WiderResNetA2
from .backbone.mobilenet import MobileNetV2
from .backbone.senet import (
    SENet,
    se_resnext50_32x4d,
    se_resnext101_32x4d,
    se_resnet50,
    se_resnet101,
    se_resnet152,
    senet154,
)

__all__ = [
    "EncoderModule",
    "ResnetEncoder",
    "SEResnetEncoder",
    "Resnet18Encoder",
    "Resnet34Encoder",
    "Resnet50Encoder",
    "Resnet101Encoder",
    "Resnet152Encoder",
    "SEResNeXt50Encoder",
    "SEResnet101Encoder",
    "SEResNeXt101Encoder",
    "SEResnet152Encoder",
    "SENet154Encoder",
    "MobilenetV2Encoder",
    "MobilenetV3Encoder",
    "SqueezenetEncoder",
    "WiderResnetEncoder",
    "WiderResnet16Encoder",
    "WiderResnet20Encoder",
    "WiderResnet38Encoder",
    "WiderResnetA2Encoder",
    "WiderResnet16A2Encoder",
    "WiderResnet38A2Encoder",
    "WiderResnet20A2Encoder",
    "DenseNetEncoder",
    "DenseNet121Encoder",
    "DenseNet169Encoder",
    "DenseNet201Encoder",
    "EfficientNetEncoder",
    "EfficientNetB0Encoder",
    "EfficientNetB1Encoder",
    "EfficientNetB2Encoder",
    "EfficientNetB3Encoder",
    "EfficientNetB4Encoder",
    "EfficientNetB5Encoder",
    "EfficientNetB6Encoder",
    "EfficientNetB7Encoder",
]


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


class ResnetEncoder(EncoderModule):
    def __init__(self, resnet, filters, strides, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__(filters, strides, layers)

        self.layer0 = nn.Sequential(
            OrderedDict(
                [("conv1", resnet.conv1), ("bn1", resnet.bn1), ("relu", resnet.relu)]
            )
        )
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
    def __init__(self, pretrained=True, layers=None):
        super().__init__(
            resnet18(pretrained=pretrained),
            [64, 64, 128, 256, 512],
            [2, 4, 8, 16, 32],
            layers,
        )


class Resnet34Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(
            resnet34(pretrained=pretrained),
            [64, 64, 128, 256, 512],
            [2, 4, 8, 16, 32],
            layers,
        )


class Resnet50Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(
            resnet50(pretrained=pretrained),
            [64, 256, 512, 1024, 2048],
            [2, 4, 8, 16, 32],
            layers,
        )


class Resnet101Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(
            resnet101(pretrained=pretrained),
            [64, 256, 512, 1024, 2048],
            [2, 4, 8, 16, 32],
            layers,
        )


class Resnet152Encoder(ResnetEncoder):
    def __init__(self, pretrained=True, layers=None):
        super().__init__(
            resnet152(pretrained=pretrained),
            [64, 256, 512, 1024, 2048],
            [2, 4, 8, 16, 32],
            layers,
        )


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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Fire(64, 16, 64, 64),
        # Fire(128, 16, 64, 64),
        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.layer1 = nn.Sequential(
            squeezenet.features[3],
            squeezenet.features[4],
            # squeezenet.features[5],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Fire(128, 32, 128, 128),
        # Fire(256, 32, 128, 128),
        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.layer2 = nn.Sequential(
            squeezenet.features[6],
            squeezenet.features[7],
            # squeezenet.features[8],
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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
    def __init__(self, layers=[2, 3, 5, 7], activation="relu6"):
        super().__init__(
            [32, 16, 24, 32, 64, 96, 160, 320], [2, 2, 4, 8, 16, 16, 32, 32], layers
        )
        encoder = MobileNetV2(activation=activation)

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
        return [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
        ]


class MobilenetV3Encoder(EncoderModule):
    def __init__(
        self, input_channels=3, small=False, drop_prob=0.0, layers=[1, 2, 3, 4]
    ):
        super().__init__(
            [24, 24, 40, 96, 96] if small else [16, 40, 80, 160, 160],
            [4, 8, 16, 32, 32],
            layers,
        )
        encoder = MobileNetV3(
            in_channels=input_channels, small=small, drop_prob=drop_prob
        )

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.act1 = encoder.act1

        self.layer0 = encoder.layer0
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        output_features = []

        x = self.layer0(x)
        output_features.append(x)

        x = self.layer1(x)
        output_features.append(x)

        x = self.layer2(x)
        output_features.append(x)

        x = self.layer3(x)
        output_features.append(x)

        x = self.layer4(x)
        output_features.append(x)

        # Return only features that were requested
        return _take(output_features, self._layers)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]


class WiderResnetEncoder(EncoderModule):
    def __init__(self, structure: List[int], layers: List[int], norm_act=ABN):
        super().__init__(
            [64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32, 32], layers
        )

        encoder = WiderResNet(structure, classes=0, norm_act=norm_act)
        self.layer0 = encoder.mod1
        self.layer1 = encoder.mod2
        self.layer2 = encoder.mod3
        self.layer3 = encoder.mod4
        self.layer4 = encoder.mod5
        self.layer5 = encoder.mod6
        self.layer6 = encoder.mod7

        self.pool2 = encoder.pool2
        self.pool3 = encoder.pool3
        self.pool4 = encoder.pool4
        self.pool5 = encoder.pool5
        self.pool6 = encoder.pool6

    @property
    def encoder_layers(self):
        return [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
        ]

    def forward(self, input):
        output_features = []

        x = self.layer0(input)
        output_features.append(x)

        x = self.layer1(self.pool2(x))
        output_features.append(x)

        x = self.layer2(self.pool3(x))
        output_features.append(x)

        x = self.layer3(self.pool4(x))
        output_features.append(x)

        x = self.layer4(self.pool5(x))
        output_features.append(x)

        x = self.layer5(self.pool6(x))
        output_features.append(x)

        x = self.layer6(x)
        output_features.append(x)

        # Return only features that were requested
        return _take(output_features, self._layers)


class WiderResnet16Encoder(WiderResnetEncoder):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2, 3, 4, 5, 6]
        super().__init__(structure=[1, 1, 1, 1, 1, 1], layers=layers)


class WiderResnet20Encoder(WiderResnetEncoder):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2, 3, 4, 5, 6]
        super().__init__(structure=[1, 1, 1, 3, 1, 1], layers=layers)


class WiderResnet38Encoder(WiderResnetEncoder):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2, 3, 4, 5, 6]
        super().__init__(structure=[3, 3, 6, 3, 1, 1], layers=layers)


class WiderResnetA2Encoder(EncoderModule):
    def __init__(self, structure: List[int], layers: List[int], norm_act=ABN):
        super().__init__(
            [64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32, 32], layers
        )

        encoder = WiderResNetA2(structure=structure, classes=0, norm_act=norm_act)
        self.layer0 = encoder.mod1
        self.layer1 = encoder.mod2
        self.layer2 = encoder.mod3
        self.layer3 = encoder.mod4
        self.layer4 = encoder.mod5
        self.layer5 = encoder.mod6
        self.layer6 = encoder.mod7

        self.pool2 = encoder.pool2
        self.pool3 = encoder.pool3

    @property
    def encoder_layers(self):
        return [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
        ]

    def forward(self, input):
        output_features = []

        out = self.layer0(input)
        output_features.append(out)

        out = self.layer1(self.pool2(out))
        output_features.append(out)

        out = self.layer2(self.pool3(out))
        output_features.append(out)

        out = self.layer3(out)
        output_features.append(out)

        out = self.layer4(out)
        output_features.append(out)

        out = self.layer5(out)
        output_features.append(out)

        out = self.layer6(out)
        output_features.append(out)

        # Return only features that were requested
        return _take(output_features, self._layers)


class WiderResnet16A2Encoder(WiderResnetA2Encoder):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2, 3, 4, 5, 6]
        super().__init__(structure=[1, 1, 1, 1, 1, 1], layers=layers)


class WiderResnet20A2Encoder(WiderResnetA2Encoder):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2, 3, 4, 5, 6]
        super().__init__(structure=[1, 1, 1, 3, 1, 1], layers=layers)


class WiderResnet38A2Encoder(WiderResnetA2Encoder):
    def __init__(self, layers=None):
        if layers is None:
            layers = [2, 3, 4, 5, 6]
        super().__init__(structure=[3, 3, 6, 3, 1, 1], layers=layers)


class DenseNetEncoder(EncoderModule):
    def __init__(
        self,
        densenet: DenseNet,
        strides: List[int],
        channels: List[int],
        layers: List[int],
    ):
        super().__init__(channels, strides, layers)

        self.layer0 = nn.Sequential(
            densenet.features.conv0, densenet.features.norm0, densenet.features.relu0
        )
        self.pool0 = densenet.features.pool0

        self.layer1 = nn.Sequential(
            densenet.features.denseblock1, densenet.features.transition1
        )

        self.layer2 = nn.Sequential(
            densenet.features.denseblock2, densenet.features.transition2
        )

        self.layer3 = nn.Sequential(
            densenet.features.denseblock3, densenet.features.transition3
        )

        self.layer4 = nn.Sequential(densenet.features.denseblock4)

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
                output = self.pool0(output)
            input = output

        # Return only features that were requested
        return _take(output_features, self._layers)


class DenseNet121Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        densenet = densenet121(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 512, 1024]
        super().__init__(densenet, strides, channels, layers)


class DenseNet161Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        densenet = densenet161(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [96, 192, 384, 1056, 2208]
        super().__init__(densenet, strides, channels, layers)


class DenseNet169Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        densenet = densenet169(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 640, 1664]
        super().__init__(densenet, strides, channels, layers)


class DenseNet201Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False):
        if layers is None:
            layers = [1, 2, 3, 4]
        densenet = densenet201(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 896, 1920]
        super().__init__(densenet, strides, channels, layers)


class EfficientNetEncoder(EncoderModule):
    def __init__(self, efficientnet, filters, strides, layers):
        super().__init__(filters, strides, layers)

        self.stem = efficientnet.stem

        self.block0 = efficientnet.block0
        self.block1 = efficientnet.block1
        self.block2 = efficientnet.block2
        self.block3 = efficientnet.block3
        self.block4 = efficientnet.block4
        self.block5 = efficientnet.block5
        self.block6 = efficientnet.block6

    @property
    def encoder_layers(self):
        return [
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
        ]

    def forward(self, x):
        input = self.stem(x)

        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)
            input = output

        # Return only features that were requested
        return _take(output_features, self._layers)


class EfficientNetB0Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b0(num_classes=1, **kwargs),
            [16, 24, 40, 80, 112, 192, 320],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB1Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b1(num_classes=1),
            [16, 24, 40, 80, 112, 192, 320],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB2Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b2(num_classes=1),
            [16, 24, 48, 88, 120, 208, 352],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB3Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b3(num_classes=1),
            [24, 32, 48, 96, 136, 232, 384],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB4Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b4(num_classes=1),
            [24, 32, 56, 112, 160, 272, 448],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB5Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b5(num_classes=1),
            [24, 40, 64, 128, 176, 304, 512],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB6Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b6(num_classes=1),
            [32, 40, 72, 144, 200, 344, 576],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB7Encoder(EfficientNetEncoder):
    def __init__(self, layers=[1, 2, 4, 6], **kwargs):
        super().__init__(
            efficient_net_b7(num_classes=1),
            [32, 48, 80, 160, 224, 384, 640],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )
