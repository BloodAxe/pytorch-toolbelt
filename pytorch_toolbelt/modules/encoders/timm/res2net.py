from .common import GenericTimmEncoder
from ..common import make_n_channel_input

__all__ = ["TimmRes2Net101Encoder", "TimmRes2Next50Encoder"]


class TimmRes2Net101Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models.res2net import res2net101_26w_4s

        encoder = res2net101_26w_4s(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class TimmRes2Next50Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None):
        from timm.models.res2net import res2next50

        encoder = res2next50(pretrained=pretrained, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self
