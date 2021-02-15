from ..common import EncoderModule, make_n_channel_input, _take


__all__ = [
    "NFNetF0SEncoder",
    "NFNetF1SEncoder",
    "NFNetF2SEncoder",
    "NFNetF3SEncoder",
    "NFNetF4SEncoder",
    "NFNetF5SEncoder",
    "NFNetF6SEncoder",
    "NFNetF7SEncoder",
]


class NFNetF0SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f0s(pretrained=pretrained, features_only=True)
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


class NFNetF1SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f1s(pretrained=pretrained, features_only=True)
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


class NFNetF2SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f2s(pretrained=pretrained, features_only=True)
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


class NFNetF3SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f3s(pretrained=pretrained, features_only=True)
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


class NFNetF4SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f4s(pretrained=pretrained, features_only=True)
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


class NFNetF5SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f5s(pretrained=pretrained, features_only=True)
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


class NFNetF6SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f6s(pretrained=pretrained, features_only=True)
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


class NFNetF7SEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f7s(pretrained=pretrained, features_only=True)
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
