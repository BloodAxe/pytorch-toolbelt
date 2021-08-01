from collections import OrderedDict
from typing import List, Tuple, Union, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .common import DecoderModule
from ..dsconv import DepthwiseSeparableConv2d
from ..activations import instantiate_activation_block, ACT_RELU

__all__ = ["ASPPModule", "SeparableASPPModule", "DeeplabV3Decoder", "DeeplabV3PlusDecoder"]


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
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates=(12, 24, 36),
        dropout: float = 0.5,
        activation: str = ACT_RELU,
        aspp_module=Union[Type[ASPPModule], Type[SeparableASPPModule]],
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
            instantiate_activation_block(activation, inplace=False),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        res = []
        for aspp in self.aspp:
            res.append(aspp(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeeplabV3Decoder(DecoderModule):
    """
    Implements DeepLabV3 model from `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Partially copy-pasted from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
    """

    def __init__(
        self,
        feature_maps: List[int],
        aspp_channels: int,
        channels: int,
        atrous_rates=(12, 24, 36),
        dropout: float = 0.5,
        activation=ACT_RELU,
    ):
        """

        Args:
            feature_maps: List of input channels
            aspp_channels:
            channels: Output channels
            atrous_rates:
            dropout:
            activation:
        """
        super().__init__()
        self.aspp = ASPP(
            in_channels=feature_maps[-1],
            out_channels=aspp_channels,
            atrous_rates=atrous_rates,
            dropout=dropout,
            activation=activation,
        )
        self.final = nn.Sequential(
            nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(aspp_channels, channels, kernel_size=1),
        )

        self._channels = [channels]
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        high_level_features = feature_maps[-1]
        high_level_features = self.aspp(high_level_features)
        return self.final(high_level_features)

    @property
    def channels(self) -> Tuple[int]:
        return self._channels


class DeeplabV3PlusDecoder(DecoderModule):
    """
    Implements DeepLabV3 model from `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Partially copy-pasted from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
    """

    def __init__(
        self,
        feature_maps: List[int],
        aspp_channels: int,
        channels: int,
        atrous_rates=(12, 24, 36),
        dropout: float = 0.5,
        activation: str = ACT_RELU,
        low_level_channels: int = 48,
    ):
        """

        Args:
            feature_maps: Input feature maps
            aspp_channels:
            channels: Number of output channels
            atrous_rates:
            dropout:
            activation:
            low_level_channels:
        """
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(feature_maps[0], low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            instantiate_activation_block(activation, inplace=True),
        )

        self.aspp = ASPP(
            in_channels=feature_maps[-1],
            out_channels=aspp_channels,
            atrous_rates=atrous_rates,
            dropout=dropout,
            activation=activation,
            aspp_module=SeparableASPPModule,
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                aspp_channels + low_level_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            instantiate_activation_block(activation, inplace=True),
        )
        self._channels = [channels]
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        low_level_features = self.project(feature_maps[0])
        output_feature = self.aspp(feature_maps[-1])

        high_level_features = F.interpolate(
            output_feature, size=low_level_features.shape[2:], mode="bilinear", align_corners=False
        )
        combined_features = torch.cat([low_level_features, high_level_features], dim=1)
        return [self.final(combined_features)]

    @property
    def channels(self) -> Tuple[int]:
        return self._channels
