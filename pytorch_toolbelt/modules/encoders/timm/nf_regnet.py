from ..common import EncoderModule, make_n_channel_input, _take


__all__ = [
    "NFRegNetB0Encoder",
    "NFRegNetB1Encoder",
    "NFRegNetB2Encoder",
    "NFRegNetB3Encoder",
    "NFRegNetB4Encoder",
    "NFRegNetB5Encoder",
]


class NFRegNetB0Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nf_regnet_b0(pretrained=pretrained, features_only=True)
        super().__init__([40, 40, 80, 160, 960], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB1Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nf_regnet_b1(pretrained=pretrained, features_only=True)
        super().__init__([40, 40, 80, 160, 960], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB2Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [0, 1, 2, 3, 4]
        encoder = nfnet.nf_regnet_b2(pretrained=pretrained, features_only=True)
        super().__init__([40, 40, 88, 176, 1064], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB3Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [0, 1, 2, 3, 4]
        encoder = nfnet.nf_regnet_b3(pretrained=pretrained, features_only=True)
        super().__init__([40, 40, 96, 184, 1152], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB4Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [0, 1, 2, 3, 4]
        encoder = nfnet.nf_regnet_b4(pretrained=pretrained, features_only=True)
        super().__init__([48, 48, 112, 216, 1344], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB5Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [0, 1, 2, 3, 4]
        encoder = nfnet.nf_regnet_b5(pretrained=pretrained, features_only=True)
        super().__init__([64, 64, 128, 256, 1536], [2, 4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return _take(y, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self
