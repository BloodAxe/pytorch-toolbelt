import torch.nn.init
from torch import nn

__all__ = ["conv1x1", "conv3x3"]


def conv1x1(in_channels: int, out_channels: int, groups=1, bias=True) -> nn.Conv2d:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
    if bias:
        torch.nn.init.zeros_(conv.bias)
    return conv


def conv3x3(in_channels: int, out_channels: int, stride=1, groups=1, bias=True) -> nn.Conv2d:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)
    if bias:
        torch.nn.init.zeros_(conv.bias)
    return conv
