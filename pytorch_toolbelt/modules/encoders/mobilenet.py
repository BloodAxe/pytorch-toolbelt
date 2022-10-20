from .common import EncoderModule, _take, make_n_channel_input
from ..backbone.mobilenet import MobileNetV2

__all__ = ["MobilenetV2Encoder", "MobileNetV3Large", "MobileNetV3Small"]


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

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self


class MobileNetV3Large(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from torchvision.models.mobilenetv3 import mobilenet_v3_large

        encoder = mobilenet_v3_large(pretrained=pretrained)

        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__([16, 24, 40, 112, 960], [2, 4, 8, 16, 32], layers)
        from torchvision.models import MobileNetV3

        self.layer0 = encoder.features[0:2]
        self.layer1 = encoder.features[2:4]
        self.layer2 = encoder.features[4:7]
        self.layer3 = encoder.features[7:13]
        self.layer4 = encoder.features[13:]

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.layer0[0][0] = make_n_channel_input(self.layer0[0][0], input_channels, mode=mode, **kwargs)
        return self


class MobileNetV3Small(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from torchvision.models.mobilenetv3 import mobilenet_v3_small

        encoder = mobilenet_v3_small(pretrained=pretrained)

        if layers is None:
            layers = [1, 2, 3, 4]
        super().__init__([16, 16, 24, 48, 576], [2, 4, 8, 16, 32], layers)
        from torchvision.models import MobileNetV3

        self.layer0 = encoder.features[0:1]
        self.layer1 = encoder.features[1:2]
        self.layer2 = encoder.features[2:4]
        self.layer3 = encoder.features[4:9]
        self.layer4 = encoder.features[9:]

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.layer0[0][0] = make_n_channel_input(self.layer0[0][0], input_channels, mode=mode, **kwargs)
        return self
