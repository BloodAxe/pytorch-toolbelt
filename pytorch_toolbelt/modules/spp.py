from collections import OrderedDict
from typing import Union, Type

import torch
from torch import nn, Tensor

from pytorch_toolbelt.modules import ACT_RELU, instantiate_activation_block, DepthwiseSeparableConv2d


class ASPPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_layer=nn.BatchNorm2d,
        activation=ACT_RELU,
    ):
        super(ASPPModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.abn = nn.Sequential(
            OrderedDict(
                [("norm", norm_layer(out_channels)), ("act", instantiate_activation_block(activation, inplace=True))]
            )
        )

    def forward(self, x):  # skipcq: PYL-W0221
        x = self.conv(x)
        x = self.abn(x)
        return x


class SeparableASPPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_layer=nn.BatchNorm2d,
        activation=ACT_RELU,
    ):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.abn = nn.Sequential(
            OrderedDict(
                [("norm", norm_layer(out_channels)), ("act", instantiate_activation_block(activation, inplace=True))]
            )
        )

    def forward(self, x):  # skipcq: PYL-W0221
        x = self.conv(x)
        x = self.abn(x)
        return x


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, norm_layer=nn.BatchNorm2d, activation: str = ACT_RELU):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.abn = nn.Sequential(
            OrderedDict(
                [("norm", norm_layer(out_channels)), ("act", instantiate_activation_block(activation, inplace=True))]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        x = self.pooling(x)
        x = self.conv(x)
        x = self.abn(x)
        return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aspp_module: Union[Type[ASPPModule], Type[SeparableASPPModule]],
        atrous_rates=(12, 24, 36),
        dropout: float = 0.5,
        activation: str = ACT_RELU,
    ):
        super(ASPP, self).__init__()
        aspp_modules = [
            aspp_module(in_channels, out_channels, 3, padding=1, dilation=1, activation=activation),
            ASPPPooling(in_channels, out_channels),
        ] + [aspp_module(in_channels, out_channels, 3, padding=ar, dilation=ar) for ar in atrous_rates]

        self.aspp = nn.ModuleList(aspp_modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.aspp) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Dropout2d(dropout, inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        res = []
        for aspp in self.aspp:
            res.append(aspp(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
