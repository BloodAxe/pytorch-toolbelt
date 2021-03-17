from .common import GenericTimmEncoder, make_n_channel_input_std_conv

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


class NFNetF0Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f0(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF1Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f1(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF2Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f2(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF3Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f3(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF4Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f4(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF5Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f5(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF6Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f6(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF7Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f7(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self
