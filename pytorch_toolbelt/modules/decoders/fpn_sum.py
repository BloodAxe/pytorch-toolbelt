from functools import partial
from itertools import repeat
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import SegmentationDecoderModule
from ..activations import ABN
from ..identity import Identity

__all__ = ["FPNSumDecoder", "FPNSumDecoderBlock", "FPNSumCenterBlock"]


class FPNSumCenterBlock(nn.Module):
    def __init__(
        self,
        encoder_features: int,
        decoder_features: int,
        dsv_channels: Optional[int] = None,
        abn_block=ABN,
        dropout=0.0,
    ):
        """
        Center FPN block that aggregates multi-scale context using strided average poolings

        Args:
            encoder_features: Number of input features
            decoder_features: Number of output features
            dsv_channels: Number of output features for deep supervision (usually number of channels in final mask)
            abn_block: Block for Activation + BatchNorm2d
            dropout: Dropout rate after context fusion
        """
        super().__init__()
        self.bottleneck = nn.Conv2d(encoder_features, encoder_features // 2, kernel_size=1)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.proj2 = nn.Conv2d(encoder_features // 2, encoder_features // 8, kernel_size=1)

        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.proj4 = nn.Conv2d(encoder_features // 2, encoder_features // 8, kernel_size=1)

        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.proj8 = nn.Conv2d(encoder_features // 2, encoder_features // 8, kernel_size=1)

        self.blend = nn.Conv2d(encoder_features // 2 + 3 * encoder_features // 8, decoder_features, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        self.conv1 = nn.Conv2d(decoder_features, decoder_features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(decoder_features)

        if dsv_channels is not None:
            self.dsv = nn.Conv2d(decoder_features, dsv_channels, kernel_size=1)
        else:
            self.dsv = None

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.bottleneck(x)

        p2 = self.proj2(self.pool2(x))
        p4 = self.proj4(self.pool4(x))
        p8 = self.proj8(self.pool8(x))

        x_size = x.size()[2:]
        x = torch.cat(
            [
                x,
                F.interpolate(p2, size=x_size, mode="nearest"),
                F.interpolate(p4, size=x_size, mode="nearest"),
                F.interpolate(p8, size=x_size, mode="nearest"),
            ],
            dim=1,
        )

        x = self.blend(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.abn1(x)

        if self.dsv is not None:
            dsv = self.dsv(x)
            return x, dsv

        return x


class FPNSumDecoderBlock(nn.Module):
    def __init__(
        self,
        encoder_features: int,
        decoder_features: int,
        output_features: int,
        dsv_channels: Optional[int] = None,
        abn_block=ABN,
        dropout=0.0,
    ):
        """

        Args:
            encoder_features:
            decoder_features:
            output_features:
            dsv_channels:
            abn_block:
            dropout:
        """
        super().__init__()
        self.skip = nn.Conv2d(encoder_features, decoder_features, kernel_size=1)
        if decoder_features == output_features:
            self.reduction = Identity()
        else:
            self.reduction = nn.Conv2d(decoder_features, output_features, kernel_size=1)

        self.conv1 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(output_features)
        self.drop1 = nn.Dropout2d(dropout, inplace=True)

        if dsv_channels is not None:
            self.dsv = nn.Conv2d(decoder_features, dsv_channels, kernel_size=1)
        else:
            self.dsv = None

    def forward(self, decoder_fm: Tensor, encoder_fm: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        decoder_fm = F.interpolate(decoder_fm, size=encoder_fm.size()[2:], mode="nearest")

        encoder_fm = self.skip(encoder_fm)
        x = decoder_fm + encoder_fm

        x = self.reduction(x)

        x = self.conv1(x)
        x = self.abn1(x)
        x = self.drop1(x)

        if self.dsv is not None:
            dsv = self.dsv(x)
            return x, dsv

        return x


class FPNSumDecoder(SegmentationDecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        output_channels: int,
        dsv_channels: Optional[int] = None,
        fpn_channels=256,
        dropout=0.0,
        abn_block=ABN,
        center_block=FPNSumCenterBlock,
        decoder_block=FPNSumDecoderBlock,
        final_block=partial(nn.Conv2d, kernel_size=1),
    ):
        """

        Args:
            feature_maps:
            output_channels:
            dsv_channels:
            fpn_channels:
            dropout:
            abn_block:
            center_block:
            decoder_block:
            final_block:
        """
        super().__init__()

        self.center = center_block(
            feature_maps[-1], fpn_channels, dsv_channels=dsv_channels, dropout=dropout, abn_block=abn_block
        )

        self.fpn_modules = nn.ModuleList(
            [
                decoder_block(
                    encoder_fm, decoder_fm, decoder_fm, dsv_channels=dsv_channels, dropout=dropout, abn_block=abn_block
                )
                for decoder_fm, encoder_fm in zip(repeat(fpn_channels), reversed(feature_maps[:-1]))
            ]
        )

        self.final_block = final_block(fpn_channels, output_channels)
        self.dsv_channels = dsv_channels

    def forward(self, feature_maps: List[Tensor]) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        last_feature_map = feature_maps[-1]
        feature_maps = reversed(feature_maps[:-1])

        dsv_masks = []

        output = self.center(last_feature_map)

        if self.dsv_channels:
            x, dsv = output
            dsv_masks.append(dsv)
        else:
            x = output

        for fpn_block, encoder_fm in zip(self.fpn_modules, feature_maps):
            output = fpn_block(x, encoder_fm)

            if self.dsv_channels:
                x, dsv = output
                dsv_masks.append(dsv)
            else:
                x = output

        x = self.final_block(x)

        if self.dsv_channels:
            return x, dsv_masks

        return x
