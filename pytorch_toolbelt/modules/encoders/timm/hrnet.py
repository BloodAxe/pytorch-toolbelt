from ..common import EncoderModule, make_n_channel_input, _take

__all__ = [
    "HRNetW18Encoder",
    "HRNetW32Encoder",
    "HRNetW48Encoder",
]


class HRNetW18Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import hrnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = hrnet.hrnet_w18(pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        super().__init__([64, 128, 256, 512, 1024], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class HRNetW32Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import hrnet

        if layers is None:
            layers = [1, 2, 3, 4]

        encoder = hrnet.hrnet_w32(pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        super().__init__([64, 128, 256, 512, 1024], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class HRNetW48Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import hrnet

        if layers is None:
            layers = [1, 2, 3, 4]

        encoder = hrnet.hrnet_w48(pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        super().__init__([64, 128, 256, 512, 1024], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self
