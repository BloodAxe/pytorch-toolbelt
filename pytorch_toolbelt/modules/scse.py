"""Implementation of the CoordConv modules from "Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks"

Original paper: https://arxiv.org/abs/1803.02579
"""

from torch import nn, Tensor
from torch.nn import functional as F


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze module
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x


class SpatialGate2d(nn.Module):
    """
    Spatial squeeze module
    """

    def __init__(self, channels, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.expand = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x
        x = self.avg_pool(x)
        x = self.squeeze(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2d(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation
    """

    def __init__(self, channels):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2d(channels)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)


class SpatialGate2dV2(nn.Module):
    """
    Spatial Squeeze and Channel Excitation module
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(channels // reduction, channels // reduction, kernel_size=7, dilation=3, padding=3 * 3)
        self.expand = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor):
        module_input = x

        x = self.squeeze(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.expand(x)
        x = x.sigmoid()
        return module_input * x


class ChannelSpatialGate2dV2(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2dV2(channels, reduction)

    def forward(self, x):
        return self.channel_gate(x) + self.spatial_gate(x)
