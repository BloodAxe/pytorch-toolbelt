from pytorch_toolbelt.modules.backbone.hrnet import HRNetV2

from .common import EncoderModule, make_n_channel_input

__all__ = ["HRNetV2Encoder48", "HRNetV2Encoder18", "HRNetV2Encoder34"]


class HRNetV2Encoder18(EncoderModule):
    def __init__(self, pretrained=False):
        super().__init__([144 + 72 + 36 + 18], [4], [0])
        self.hrnet = HRNetV2(width=18, pretrained=False)

    def forward(self, x):
        return self.hrnet(x)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.hrnet.layer0.conv1 = make_n_channel_input(self.hrnet.layer0.conv1, input_channels, mode)


class HRNetV2Encoder34(EncoderModule):
    def __init__(self, pretrained=False):
        super().__init__([34 * 8 + 34 * 4 + 34 * 2 + 34], [4], [0])
        self.hrnet = HRNetV2(width=34, pretrained=False)

    def forward(self, x):
        return self.hrnet(x)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.hrnet.layer0.conv1 = make_n_channel_input(self.hrnet.layer0.conv1, input_channels, mode)


class HRNetV2Encoder48(EncoderModule):
    def __init__(self, pretrained=False):
        super().__init__([720], [4], [0])
        self.hrnet = HRNetV2(width=48, pretrained=False)

    def forward(self, x):
        return self.hrnet(x)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.hrnet.layer0.conv1 = make_n_channel_input(self.hrnet.layer0.conv1, input_channels, mode)
