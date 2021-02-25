from __future__ import absolute_import

from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..modules.activations import ABN

__all__ = ["FPNContextBlock", "FPNBottleneckBlock", "FPNFuse", "FPNFuseSum", "HFF"]


class FPNContextBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN, dropout=0.0):
        """
        Center FPN block that aggregates multi-scale context using strided average poolings

        :param in_channels: Number of input features
        :param out_channels: Number of output features
        :param abn_block: Block for Activation + BatchNorm2d
        :param dropout: Dropout rate after context fusion
        """
        super().__init__()
        self.bottleneck = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)

        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.proj4 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)

        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.proj8 = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)

        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.proj_global = nn.Conv2d(in_channels // 2, in_channels // 8, kernel_size=1)

        self.blend = nn.Conv2d(4 * in_channels // 8, out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_channels)

        self.dropout = nn.Dropout2d(dropout, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.bottleneck(x)

        p2 = self.proj2(self.pool2(x))
        p4 = self.proj4(self.pool4(x))
        p8 = self.proj8(self.pool8(x))
        pg = self.proj_global(self.pool_global(x))

        out_size = p2.size()[2:]

        x = torch.cat(
            [
                p2,
                F.interpolate(p4, size=out_size, mode="nearest"),
                F.interpolate(p8, size=out_size, mode="nearest"),
                F.interpolate(pg, size=out_size, mode="nearest"),
            ],
            dim=1,
        )

        x = self.blend(x)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class FPNBottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN, dropout=0.0):
        """

        Args:
            encoder_features:
            decoder_features:
            output_features:
            supervision_channels:
            abn_block:
            dropout:
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_channels)

        self.drop1 = nn.Dropout2d(dropout, inplace=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221

        x = self.conv1(x)
        x = self.abn1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.abn2(x)
        return x


class FPNFuse(nn.Module):
    def __init__(self, mode="bilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: List[Tensor]):  # skipcq: PYL-W0221
        layers = []
        dst_size = features[0].size()[2:]  # Skip B, C, and use rest (This works for 1D, 2D, 3D and ND..)

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)


class FPNFuseSum(nn.Module):
    """Compute a sum of individual FPN layers"""

    def __init__(self, mode="bilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: List[Tensor]) -> Tensor:  # skipcq: PYL-W0221
        output = features[0]
        dst_size = features[0].size()[2:]  # Skip B, C, and use rest (This works for 1D, 2D, 3D and ND..)

        for f in features[1:]:
            output = output + F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners)

        return output


class HFF(nn.Module):
    """
    Hierarchical feature fusion module.
    https://arxiv.org/pdf/1811.11431.pdf
    https://arxiv.org/pdf/1803.06815.pdf

    What it does is easily explained in code:
    feature_map_0 - feature_map of the highest resolution
    feature_map_N - feature_map of the smallest resolution

    >>> feature_map = feature_map_0 + up(feature_map[1] + up(feature_map[2] + up(feature_map[3] + ...))))
    """

    def __init__(self, sizes=None, upsample_scale=2, mode="nearest", align_corners=None):
        super().__init__()
        self.sizes = sizes
        self.interpolation_mode = mode
        self.align_corners = align_corners
        self.upsample_scale = upsample_scale

    def forward(self, features: List[Tensor]) -> Tensor:  # skipcq: PYL-W0221
        num_feature_maps = len(features)

        current_map = features[-1]
        for feature_map_index in reversed(range(num_feature_maps - 1)):
            if self.sizes is not None:
                prev_upsampled = self._upsample(current_map, self.sizes[feature_map_index])
            else:
                prev_upsampled = self._upsample(current_map)

            current_map = features[feature_map_index] + prev_upsampled

        return current_map

    def _upsample(self, x, output_size=None):
        if output_size is not None:
            x = F.interpolate(
                x,
                size=(output_size[0], output_size[1]),
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
        else:
            x = F.interpolate(
                x, scale_factor=self.upsample_scale, mode=self.interpolation_mode, align_corners=self.align_corners
            )
        return x
