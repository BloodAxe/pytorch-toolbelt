from ..common import EncoderModule, make_n_channel_input

__all__ = [
    "DPN68BEncoder",
    "DPN68Encoder",
    "DPN92Encoder",
    "DPN107Encoder",
    "DPN131Encoder",
]


class DPN68Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn68(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([144, 320, 704, 832], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN68BEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn68b(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([144, 320, 704, 832], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN92Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn92(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([336, 704, 1552, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN107Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]

        encoder = dpn.dpn107(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([376, 1152, 2432, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self


class DPN131Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import dpn

        if layers is None:
            layers = [0, 1, 2, 3]
        encoder = dpn.dpn131(pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        super().__init__([352, 832, 1984, 2688], [4, 8, 16, 32], layers)
        self.encoder = encoder

    def forward(self, x):
        y = self.encoder.forward(x)
        return y

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.features_conv1_1.conv = make_n_channel_input(
            self.encoder.features_conv1_1.conv, input_channels, mode, **kwargs
        )
        return self
