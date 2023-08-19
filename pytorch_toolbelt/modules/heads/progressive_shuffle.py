from typing import Optional, Callable

import numpy as np
from torch import nn

from pytorch_toolbelt.modules import instantiate_activation_block
from pytorch_toolbelt.modules.interfaces import AbstractHead, FeatureMapsSpecification

__all__ = ["ProgressiveShuffleHead"]


class ProgressiveShuffleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, padding=0, bias=False),
        )

        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


def divisible(channels: int, divisor: int) -> int:
    return int(np.ceil(channels / float(divisor))) * divisor


class ProgressiveShuffleHead(AbstractHead):
    """
    Progressive Shuffle head takes the largest feature map and progressively upsample it to the original image size.
    Upsampling is done by doing [Conv3x3 + BN + Activation + Conv1x1 + PixelShuffle].
    At each stage number of channels is reduced by a factor given by `reduction_factor`.

    """

    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        activation: str,
        dropout_rate,
        output_name: Optional[str],
        reduction_factor: int = 2,
    ):
        super().__init__(input_spec)

        self.num_classes = num_classes
        self.feature_map_index = input_spec.get_index_of_largest_feature_map()
        self.output_name = output_name

        min_stride = input_spec.strides[self.feature_map_index]
        num_upsample_blocks = int(np.log2(min_stride))
        in_channels = input_spec.channels[self.feature_map_index]
        blocks = []

        for _ in range(num_upsample_blocks):
            out_channels = divisible(in_channels / reduction_factor, 8)
            blocks.append(ProgressiveShuffleBlock(in_channels, out_channels, activation=activation))
            in_channels = out_channels

        blocks += [
            nn.Dropout2d(p=dropout_rate),
        ]

        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Conv2d(in_channels, self.num_classes, kernel_size=3, padding=1, bias=True)

        self.output_spec = FeatureMapsSpecification(channels=(num_classes,), strides=(1,))

    def forward(self, feature_maps, output_size=None):
        x = self.blocks(feature_maps[self.feature_map_index])
        output = self.final(x)

        if self.output_name is not None:
            return {self.output_name: output}
        else:
            return output

    def apply_to_final_layer(self, func: Callable[[nn.Module], None]):
        func(self.final)
