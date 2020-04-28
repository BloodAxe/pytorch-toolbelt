from collections import OrderedDict
from functools import partial
from typing import Union

from torch import nn

from .common import EncoderModule, make_n_channel_input
from ..unet import UnetBlock

__all__ = ["UnetEncoder"]


class UnetEncoder(EncoderModule):
    """
    Vanilla U-Net encoder
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        num_layers=4,
        growth_factor=2,
        pool_block: Union[nn.MaxPool2d, nn.AvgPool2d] = None,
        unet_block: Union[nn.Module, UnetBlock] = UnetBlock,
    ):
        if pool_block is None:
            pool_block = partial(nn.MaxPool2d, kernel_size=2, stride=2)

        feature_maps = [out_channels * (growth_factor ** i) for i in range(num_layers)]
        strides = [2 ** i for i in range(num_layers)]
        super().__init__(feature_maps, strides, layers=list(range(num_layers)))

        input_filters = in_channels
        self.num_layers = num_layers
        for layer in range(num_layers):
            block = unet_block(input_filters, feature_maps[layer])

            if layer > 0:
                pool = pool_block()
                block = nn.Sequential(OrderedDict([("pool", pool), ("conv", block)]))

            input_filters = feature_maps[layer]
            self.add_module(f"layer{layer}", block)

    @property
    def encoder_layers(self):
        return [self.__getattr__(f"layer{layer}") for layer in range(self.num_layers)]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0.conv1 = make_n_channel_input(self.layer0.conv1, input_channels, mode)
        return self
