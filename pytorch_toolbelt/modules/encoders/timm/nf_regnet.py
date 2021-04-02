from .common import GenericTimmEncoder, make_n_channel_input_std_conv


__all__ = [
    "NFRegNetB0Encoder",
    "NFRegNetB1Encoder",
    "NFRegNetB2Encoder",
    "NFRegNetB3Encoder",
    "NFRegNetB4Encoder",
    "NFRegNetB5Encoder",
]


class NFRegNetB0Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nf_regnet_b0(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv = make_n_channel_input_std_conv(
            self.encoder.stem_conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB1Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nf_regnet_b1(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv = make_n_channel_input_std_conv(
            self.encoder.stem_conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB2Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nf_regnet_b2(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv = make_n_channel_input_std_conv(
            self.encoder.stem_conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB3Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nf_regnet_b3(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv = make_n_channel_input_std_conv(
            self.encoder.stem_conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB4Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nf_regnet_b4(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv = make_n_channel_input_std_conv(
            self.encoder.stem_conv, input_channels, mode, **kwargs
        )
        return self


class NFRegNetB5Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models import nfnet

        encoder = nfnet.nf_regnet_b5(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.stem_conv = make_n_channel_input_std_conv(
            self.encoder.stem_conv, input_channels, mode, **kwargs
        )
        return self
