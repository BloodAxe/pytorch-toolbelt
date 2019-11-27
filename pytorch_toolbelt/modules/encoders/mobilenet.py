from .common import EncoderModule, _take, make_n_channel_input
from ..backbone.mobilenet import MobileNetV2
from ..backbone.mobilenetv3 import MobileNetV3

__all__ = ["MobilenetV2Encoder", "MobilenetV3Encoder"]


class MobilenetV2Encoder(EncoderModule):
    def __init__(self, layers=[2, 3, 5, 7], activation="relu6"):
        super().__init__([32, 16, 24, 32, 64, 96, 160, 320], [2, 2, 4, 8, 16, 16, 32, 32], layers)
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
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


class MobilenetV3Encoder(EncoderModule):
    def __init__(self, input_channels=3, small=False, drop_prob=0.0, layers=[1, 2, 3, 4]):
        super().__init__([24, 24, 40, 96, 96] if small else [16, 40, 80, 160, 160], [4, 8, 16, 32, 32], layers)
        encoder = MobileNetV3(in_channels=input_channels, small=small, drop_prob=drop_prob)

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

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self
