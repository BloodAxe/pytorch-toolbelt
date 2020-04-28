from torch import nn

__all__ = ["conv1x1"]


def conv1x1(in_channels: int, out_channels: int, groups=1, bias=True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
