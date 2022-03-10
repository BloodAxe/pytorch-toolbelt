from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .common import DecoderModule
from ..activations import get_activation_block, ACT_RELU

__all__ = ["BiFPNDecoder", "BiFPNBlock", "BiFPNConvBlock", "BiFPNDepthwiseConvBlock"]


class BiFPNDepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.


    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, act=nn.ReLU):
        super(BiFPNDepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = act(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=nn.ReLU, dilation=1):
        super(BiFPNConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = act(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size: int = 64, epsilon: float = 0.0001, act=nn.ReLU):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p4_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p5_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p6_td = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)

        self.p4_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p5_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p6_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)
        self.p7_out = BiFPNDepthwiseConvBlock(feature_size, feature_size, act=act)

        self.w1 = nn.Parameter(torch.Tensor(2, 4), requires_grad=True)
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 4), requires_grad=True)
        self.w2_relu = nn.ReLU()

        torch.nn.init.constant_(self.w1, 1)
        torch.nn.init.constant_(self.w2, 1)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs

        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)

        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, size=p6_x.size()[2:]))
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, size=p5_x.size()[2:]))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, size=p4_x.size()[2:]))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, size=p3_x.size()[2:]))

        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(
            w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * F.interpolate(p3_out, size=p4_x.size()[2:])
        )
        p5_out = self.p5_out(
            w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * F.interpolate(p4_out, size=p5_x.size()[2:])
        )
        p6_out = self.p6_out(
            w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * F.interpolate(p5_out, size=p6_x.size()[2:])
        )
        p7_out = self.p7_out(
            w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * F.interpolate(p6_out, size=p7_x.size()[2:])
        )

        return [p3_out, p4_out, p5_out, p6_out, p7_out]


class BiFPNDecoder(DecoderModule):
    """
    BiFPN decoder

    Expects input of three feature maps

    Reference: https://arxiv.org/abs/1911.09070
    """

    def __init__(
        self, feature_maps: List[int], strides: List[int], channels: int = 64, num_layers: int = 2, activation=ACT_RELU
    ):
        super(BiFPNDecoder, self).__init__()
        act = get_activation_block(activation)
        if len(feature_maps) != 3:
            raise ValueError("Number of input feature maps must be equal 3")
        self.p3 = nn.Conv2d(feature_maps[0], channels, kernel_size=(1, 1))
        self.p4 = nn.Conv2d(feature_maps[1], channels, kernel_size=(1, 1))
        self.p5 = nn.Conv2d(feature_maps[2], channels, kernel_size=(1, 1))

        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(feature_maps[2], channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = BiFPNConvBlock(channels, channels, kernel_size=3, stride=2, padding=1, act=act)

        bifpns: List[BiFPNBlock] = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(channels, act=act))
        self.bifpn = nn.ModuleList(bifpns)

        self._channels = [channels] * 5
        self._strides = tuple(strides) + (strides[-1] * 2, strides[-1] * 4)

    @property
    def channels(self):
        return self._channels

    @property
    def strides(self):
        return self._strides

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        c3, c4, c5 = inputs

        # Calculate the input column of BiFPN
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(c5)
        p7_x = self.p7(p6_x)

        features = p3_x, p4_x, p5_x, p6_x, p7_x
        for bifpn in self.bifpn:
            features = bifpn(features)
        return features
