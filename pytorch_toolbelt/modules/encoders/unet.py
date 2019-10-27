from torch import nn


from ..abn import ABN

from .common import EncoderModule, _take

__all__ = ["UnetEncoderBlock", "UnetEncoder"]


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, abn_block=ABN, stride=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False, **kwargs)
        self.bn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=stride, bias=False, **kwargs)
        self.bn2 = abn_block(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UnetEncoder(EncoderModule):
    def __init__(self, input_channels=3, features=32, num_layers=4, growth_factor=2, abn_block=ABN):
        feature_maps = [features * growth_factor * (i + 1) for i in range(num_layers)]
        strides = [2 * (i + 1) for i in range(num_layers)]
        super().__init__(feature_maps, strides, layers=list(range(num_layers)))

        input_filters = input_channels
        output_filters = feature_maps[0]
        self.num_layers = num_layers
        for layer in range(num_layers):
            block = UnetEncoderBlock(input_filters, output_filters, abn_block=abn_block)

            self.add_module(f"layer{layer}", block)

    @property
    def encoder_layers(self):
        return [self[f"layer{layer}"] for layer in range(self.num_layers)]
