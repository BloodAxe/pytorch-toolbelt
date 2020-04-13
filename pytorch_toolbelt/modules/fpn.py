from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F

from ..modules.activations import ABN

__all__ = [
    "FPNBottleneckBlock",
    "FPNBottleneckBlockBN",
    "FPNPredictionBlock",
    "FPNFuse",
    "FPNFuseSum",
    "FPNFinalBottleneckBlock",
    "FPNFinalTransposeConvBlock",
    "UpsampleAdd",
    "UpsampleAddConv",
]


class FPNBottleneckBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FPNBottleneckBlockBN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FPNPredictionBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FPNFinalBottleneckBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, reduction=4, abn_block=ABN):
        super().__init__()

        features = input_channels // reduction

        self.bottleneck = nn.Conv2d(input_channels, features, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(features)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(features)

        self.final = nn.Conv2d(features, output_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.bottleneck(x)

        x = self.conv1(x)
        x = self.abn1(x)

        x = self.conv2(x)
        x = self.abn2(x)

        x = self.final(x)
        return x


class FPNFinalTransposeConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, reduction=4, abn_block=ABN):
        """

        Args:
            input_channels:
            output_channels:
            reduction:
            abn_block:
        """
        super().__init__()

        features = input_channels // reduction

        self.bottleneck = nn.Conv2d(input_channels, features, kernel_size=1, bias=False)

        self.conv1 = nn.ConvTranspose2d(features, features, kernel_size=3, stride=2, padding=1, bias=False)
        self.abn1 = abn_block(features)

        self.conv2 = nn.ConvTranspose2d(features, features, kernel_size=3, stride=2, padding=1, bias=False)
        self.abn2 = abn_block(features)

        self.final = nn.Conv2d(features, output_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.bottleneck(x)

        x = self.conv1(x)
        x = self.abn1(x)

        x = self.conv2(x)
        x = self.abn2(x)

        x = self.final(x)
        return x


class UpsampleAdd(nn.Module):
    """
    Compute pixelwise sum of first tensor and upsampled second tensor.
    """

    def __init__(self, filters: int, upsample_scale=None, mode="nearest", align_corners=None):
        super().__init__()
        self.interpolation_mode = mode
        self.upsample_scale = upsample_scale
        self.align_corners = align_corners

    def forward(self, x, y=None):
        if y is not None:
            if self.upsample_scale is not None:
                y = F.interpolate(
                    y, scale_factor=self.upsample_scale, mode=self.interpolation_mode, align_corners=self.align_corners
                )
            else:
                y = F.interpolate(
                    y, size=(x.size(2), x.size(3)), mode=self.interpolation_mode, align_corners=self.align_corners
                )

            x = x + y

        return x


class UpsampleAddConv(nn.Module):
    """
    Compute pixelwise sum of first tensor and upsampled second tensor and convolve with 3x3 kernel
    to smooth aliasing artifacts
    """

    def __init__(self, filters: int, upsample_scale=None, mode="nearest", align_corners=None):
        super().__init__()
        self.interpolation_mode = mode
        self.upsample_scale = upsample_scale
        self.align_corners = align_corners
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x, y=None):
        if y is not None:
            if self.upsample_scale is not None:
                y = F.interpolate(
                    y, scale_factor=self.upsample_scale, mode=self.interpolation_mode, align_corners=self.align_corners
                )
            else:
                y = F.interpolate(
                    y, size=(x.size(2), x.size(3)), mode=self.interpolation_mode, align_corners=self.align_corners
                )

            x = x + y

        x = self.conv(x)
        return x


class FPNFuse(nn.Module):
    def __init__(self, mode="bilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features):
        layers = []
        dst_size = features[0].size()[-2:]

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)


class FPNFuseSum(nn.Module):
    """Compute a sum of individual FPN layers"""

    def __init__(self, mode="bilinear", align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features):
        output = features[0]
        dst_size = features[0].size()[-2:]

        for f in features[1:]:
            output = output + F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners)

        return output


class HFF(nn.Module):
    """
    Hierarchical feature fusion module.
    https://arxiv.org/pdf/1811.11431.pdf
    https://arxiv.org/pdf/1803.06815.pdf

    What it does is easily explained in code:
    feature_map_N - feature_map of the smallest resolution
    feature_map_0 - feature_map of the highest resolution

    >>> feature_map = feature_map_0 + up(feature_map[1] + up(feature_map[2] + up(feature_map[3] + ...))))
    """

    def __init__(self, sizes=None, upsample_scale=2, mode="nearest", align_corners=None):
        super().__init__()
        self.sizes = sizes
        self.interpolation_mode = mode
        self.align_corners = align_corners
        self.upsample_scale = upsample_scale

    def forward(self, features):
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
