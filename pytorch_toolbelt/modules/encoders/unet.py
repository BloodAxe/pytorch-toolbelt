from typing import Union

from torch import nn

from .common import EncoderModule, make_n_channel_input
from ..activations import ABN, AGN
from ..unet import UnetEncoderBlock

__all__ = ["UnetEncoder"]


class UnetEncoder(EncoderModule):
    """
    Vanilla U-Net encoder
    """

    def __init__(
        self, input_channels=3, features=32, num_layers=4, growth_factor=2, abn_block=Union[ABN, AGN, nn.Module]
    ):
        feature_maps = [features * growth_factor * (i + 1) for i in range(num_layers)]
        strides = [2 * (i + 1) for i in range(num_layers)]
        super().__init__(feature_maps, strides, layers=list(range(num_layers)))

        input_filters = input_channels
        self.num_layers = num_layers
        for layer in range(num_layers):
            block = UnetEncoderBlock(input_filters, feature_maps[layer], abn_block=abn_block)
            input_filters = feature_maps[layer]
            self.add_module(f"layer{layer}", block)

    @property
    def encoder_layers(self):
        return [self.__getattr__(f"layer{layer}") for layer in range(self.num_layers)]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self
