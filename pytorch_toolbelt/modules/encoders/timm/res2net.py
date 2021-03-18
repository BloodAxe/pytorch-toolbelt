from .common import GenericTimmEncoder
from ..common import make_n_channel_input
from ...activations import ACT_RELU, get_activation_block

__all__ = ["TimmRes2Net101Encoder", "TimmRes2Next50Encoder"]


class TimmRes2Net101Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        from timm.models.res2net import res2net101_26w_4s

        act_layer = get_activation_block(activation)
        encoder = res2net101_26w_4s(pretrained=pretrained, act_layer=act_layer, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self


class TimmRes2Next50Encoder(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        from timm.models.res2net import res2next50

        act_layer = get_activation_block(activation)
        encoder = res2next50(pretrained=pretrained, act_layer=act_layer, features_only=True)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1 = make_n_channel_input(self.encoder.conv1, input_channels, mode, **kwargs)
        return self
