"""Implementation of the CoordConv modules from
"Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks"

Original paper: https://arxiv.org/abs/1803.02579
"""

from torch import nn, Tensor
from torch.nn import functional as F

__all__ = ["ChannelGate2d", "SpatialGate2d", "ChannelSpatialGate2d", "SpatialGate2dV2", "ChannelSpatialGate2dV2"]


class ChannelGate2d(nn.Module):
    """
    Channel Squeeze module
    """

    def __init__(self, channels):
        super().__init__()
        self.squeeze = nn.Conv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: Tensor):  # skipcq: PYL-W0221
        module_input = x
        x = self.squeeze(x)
        x = x.sigmoid()
        return module_input * x


class SpatialGate2d(nn.Module):
    """
    Spatial squeeze module
    """

    def __init__(self, channels, reduction=None, squeeze_channels=None):
        """
        Instantiate module

        :param channels: Number of input channels
        :param reduction: Reduction factor
        :param squeeze_channels: Number of channels in squeeze block.
        """
        super().__init__()
        assert reduction or squeeze_channels, "One of 'reduction' and 'squeeze_channels' must be set"
        assert not (reduction and squeeze_channels), "'reduction' and 'squeeze_channels' are mutually exclusive"

        if squeeze_channels is None:
            squeeze_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x: Tensor):  # skipcq: PYL-W0221
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

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_gate = ChannelGate2d(channels)
        self.spatial_gate = SpatialGate2d(channels, reduction=reduction)

    def forward(self, x):  # skipcq: PYL-W0221
        return self.channel_gate(x) + self.spatial_gate(x)


class SpatialGate2dV2(nn.Module):
    """
    Spatial Squeeze and Channel Excitation module
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        squeeze_channels = max(1, channels // reduction)
        self.squeeze = nn.Conv2d(channels, squeeze_channels, kernel_size=1, padding=0)
        self.conv = nn.Conv2d(squeeze_channels, squeeze_channels, kernel_size=7, dilation=3, padding=3 * 3)
        self.expand = nn.Conv2d(squeeze_channels, channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor):  # skipcq: PYL-W0221
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

    def forward(self, x):  # skipcq: PYL-W0221
        return self.channel_gate(x) + self.spatial_gate(x)
