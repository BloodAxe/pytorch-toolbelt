from typing import List, Tuple, Union, Type, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_toolbelt.modules.decoders.common import DecoderModule
from pytorch_toolbelt.modules.activations import (
    ACT_RELU,
    instantiate_activation_block,
)
from .. import conv1x1
__all__ = ["BiFPNDecoder", "BiFPNBlock", "BiFPNConvBlock", "BiFPNDepthwiseConvBlock"]


class BiFPNDepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        activation: str = ACT_RELU,
    ):
        super(BiFPNDepthwiseConvBlock, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = instantiate_activation_block(activation, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=ACT_RELU, dilation=1):
        super(BiFPNConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = instantiate_activation_block(activation, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-Directional Feature Pyramid Network
    """

    def __init__(
        self,
        feature_size: int,
        num_feature_maps: int,
        epsilon=0.0001,
        activation=ACT_RELU,
        block: Union[Type[BiFPNConvBlock], Type[BiFPNDepthwiseConvBlock]] = BiFPNConvBlock,
    ):
        super(BiFPNBlock, self).__init__()

        self.epsilon = epsilon
        num_blocks = num_feature_maps - 1

        self.top_down_blocks = nn.ModuleList(
            [block(feature_size, feature_size, activation=activation) for _ in range(num_blocks)]
        )
        self.bottom_up_blocks = nn.ModuleList(
            [block(feature_size, feature_size, activation=activation) for _ in range(num_blocks)]
        )

        self.register_parameter("w1", nn.Parameter(torch.Tensor(2, num_blocks), requires_grad=True))
        self.register_parameter("w2", nn.Parameter(torch.Tensor(3, num_blocks), requires_grad=True))

        torch.nn.init.constant_(self.w1, 1)
        torch.nn.init.constant_(self.w2, 1)

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        transition_features = self.top_down_pathway(inputs)
        return self.bottom_up_pathway(transition_features, inputs)

    def bottom_up_pathway(self, transition_features, inputs):
        # Calculate Bottom-Up Pathway
        # p3_out = p3_td
        # p4_out = self.p4_out(
        #     w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * F.interpolate(p3_out, size=p4_x.size()[2:])
        # )
        # p5_out = self.p5_out(
        #     w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * F.interpolate(p4_out, size=p5_x.size()[2:])
        # )
        # p6_out = self.p6_out(
        #     w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * F.interpolate(p5_out, size=p6_x.size()[2:])
        # )
        # p7_out = self.p7_out(
        #     w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * F.interpolate(p6_out, size=p7_x.size()[2:])
        # )

        w2 = torch.nn.functional.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        outputs = [transition_features[-1]]
        transition_reversed = transition_features[:-1][::-1]
        for i, block in enumerate(self.bottom_up_blocks):
            x = inputs[i + 1]
            td = transition_reversed[i]
            y = block(x * w2[0, i] + td * w2[1, i] + F.interpolate(outputs[-1], size=x.size()[2:]) * w2[2, i])
            outputs.append(y)

        return outputs

    def top_down_pathway(self, inputs: List[Tensor]) -> List[Tensor]:
        # p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, size=p6_x.size()[2:]))
        # p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, size=p5_x.size()[2:]))
        # p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, size=p4_x.size()[2:]))
        # p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, size=p3_x.size()[2:]))

        w1 = torch.nn.functional.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)

        features = [inputs[-1]]
        inputs_reversed = inputs[:-1][::-1]

        for i, block in enumerate(self.top_down_blocks):
            x = inputs_reversed[i]
            y = block(w1[0, i] * x + w1[1, i] * F.interpolate(features[-1], size=x.size()[2:]))
            features.append(y)

        return features


class BiFPNDecoder(DecoderModule):
    """
    BiFPN decoder. Note this class does not compute additional feature maps (p6/p7) and expects them as input.
    Because of this it supports arbitrary number of input feature maps.

    Reference: https://arxiv.org/abs/1911.09070
    """

    def __init__(
        self,
        feature_maps: List[int],
        strides: List[int],
        channels: int,
        num_layers: int,
        activation=ACT_RELU,
        block: Union[
            Type[BiFPNConvBlock], Type[BiFPNDepthwiseConvBlock], Callable[[int, int], nn.Module]
        ] = BiFPNConvBlock,
        projection_block: Callable[[int, int], nn.Module] = conv1x1,
    ):
        super(BiFPNDecoder, self).__init__()

        self.projections = nn.ModuleList(
            [
                projection_block(f, channels) if f != channels else nn.Identity()
                for f in feature_maps
            ]
        )

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(channels, num_feature_maps=len(feature_maps), activation=activation, block=block))
        self.bifpn = nn.Sequential(*bifpns)

        self._channels = [channels] * len(feature_maps)
        self._strides = tuple(strides)

    @property
    @torch.jit.ignore
    def channels(self):
        return self._channels

    @property
    @torch.jit.ignore
    def strides(self):
        return self._strides

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        # Calculate the input column of BiFPN
        features = [p(c) for p, c in zip(self.projections, feature_maps)]
        return self.bifpn(features)


if __name__ == "__main__":
    from pytorch_toolbelt.utils.torch_utils import describe_outputs

    decoder = BiFPNDecoder(feature_maps=[256, 512, 1024], strides=[8, 16, 32], channels=256, num_layers=3)
    inputs = [torch.randn(1, 256, 64, 64), torch.randn(1, 512, 32, 32), torch.randn(1, 1024, 16, 16)]
    outputs = decoder(inputs)
    print(describe_outputs(outputs))
