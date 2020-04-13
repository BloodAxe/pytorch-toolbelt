from typing import List

from .common import EncoderModule, _take, make_n_channel_input
from ..activations import ABN
from ..backbone.wider_resnet import WiderResNet, WiderResNetA2

__all__ = [
    "WiderResnetEncoder",
    "WiderResnet16A2Encoder",
    "WiderResnet16Encoder",
    "WiderResnet20Encoder",
    "WiderResnet38A2Encoder",
    "WiderResnet38Encoder",
    "WiderResnetA2Encoder",
    "WiderResnet20A2Encoder",
]


class WiderResnetEncoder(EncoderModule):
    def __init__(self, structure: List[int], layers: List[int], norm_act=ABN):
        super().__init__([64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32, 32], layers)

        encoder: WiderResNet = WiderResNet(structure, classes=0, norm_act=norm_act)
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
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]

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

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


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
        super().__init__([64, 128, 256, 512, 1024, 2048, 4096], [1, 2, 4, 8, 16, 32, 32], layers)

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
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]

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

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


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
