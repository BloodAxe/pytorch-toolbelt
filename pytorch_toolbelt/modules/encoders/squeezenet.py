from collections import OrderedDict

from torch import nn
from torchvision.models import squeezenet1_1

from .common import EncoderModule, make_n_channel_input

__all__ = ["SqueezenetEncoder"]


class SqueezenetEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=[1, 2, 3]):
        super().__init__([64, 128, 256, 512], [4, 8, 16, 16], layers)
        squeezenet = squeezenet1_1(pretrained=pretrained)

        # nn.Conv2d(3, 64, kernel_size=3, stride=2),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        self.layer0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", squeezenet.features[0]),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
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
            squeezenet.features[9], squeezenet.features[10], squeezenet.features[11], squeezenet.features[12]
        )

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self
