from .common import GenericTimmEncoder, make_n_channel_input_std_conv

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


class NFNetF0SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f0s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF1SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        if layers is None:
            layers = [1, 2, 3, 4]
        encoder = nfnet.nfnet_f1s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF2SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f2s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF3SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f3s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF4SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f4s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF5SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f5s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF6SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f6s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self


class NFNetF7SEncoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nfnet_f7s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv1 = make_n_channel_input_std_conv(
            self.encoder.stem_conv1, input_channels, mode, **kwargs
        )
        return self
