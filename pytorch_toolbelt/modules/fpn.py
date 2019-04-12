from __future__ import absolute_import
import torch

from torch import nn
from torch.nn import functional as F

from .scse import ChannelSpatialGate2dV2
from .abn import ABN, ACT_ELU, ACT_SELU


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
        return x


class FPNPredictionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, mode='nearest'):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, kernel_size=3, padding=1)
        self.mode = mode

    def forward(self, x, y=None):
        if y is not None:
            x = x + F.interpolate(y, size=x.size()[2:], mode=self.mode, align_corners=True if self.mode == 'bilinear' else None)

        x = self.conv(x)
        return x


class FPNFuse(nn.Module):
    def __init__(self, mode='bilinear', align_corners=True):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features):
        layers = []
        dst_size = features[0].size()[-2:]

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)
