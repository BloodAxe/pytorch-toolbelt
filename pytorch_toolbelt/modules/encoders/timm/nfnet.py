from ..common import EncoderModule, make_n_channel_input, _take


__all__ = [
    "NFNetF0Encoder",
    "NFNetF1Encoder",
    "NFNetF2Encoder",
    "NFNetF3Encoder",
    "NFNetF4Encoder",
    "NFNetF5Encoder",
    "NFNetF6Encoder",
    "NFNetF7Encoder",
]


class NFNetF0Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f0(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF1Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f1(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF2Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f2(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF3Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f3(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF4Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f4(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF5Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f5(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF6Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f6(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFNetF7Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f7(pretrained=pretrained, features_only=True)
        super().__init__([64, 256, 512, 1536, 3072], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self
