from itertools import repeat
from typing import List, Tuple

import torch
from ..activated_batch_norm import ABN
from ..identity import Identity
from .common import SegmentationDecoderModule


from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["FPNSumDecoder", "FPNSumDecoderBlock", "FPNSumCenterBlock"]


class FPNSumCenterBlock(nn.Module):
    def __init__(self, encoder_features: int, decoder_features: int, num_classes: int, abn_block=ABN, dropout=0.0):
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

        self.dsv = nn.Conv2d(decoder_features, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.bottleneck(x)

        p2 = self.proj2(self.pool2(x))
        p4 = self.proj4(self.pool4(x))
        p8 = self.proj8(self.pool8(x))

        x_size = x.size()[2:]
        x = torch.cat(
            [
                x,
                F.interpolate(p2, size=x_size, mode="bilinear", align_corners=False),
                F.interpolate(p4, size=x_size, mode="bilinear", align_corners=False),
                F.interpolate(p8, size=x_size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )

        x = self.blend(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.abn1(x)

        dsv = self.dsv(x)

        return x, dsv


class FPNSumDecoderBlock(nn.Module):
    def __init__(
        self,
        encoder_features: int,
        decoder_features: int,
        output_features: int,
        num_classes: int,
        abn_block=ABN,
        dropout=0.0,
    ):
        super().__init__()
        self.skip = nn.Conv2d(encoder_features, decoder_features, kernel_size=1)
        if decoder_features == output_features:
            self.reduction = Identity()
        else:
            self.reduction = nn.Conv2d(decoder_features, output_features, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.conv1 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(output_features)

        self.dsv = nn.Conv2d(output_features, num_classes, kernel_size=1)

    def forward(self, decoder_fm: Tensor, encoder_fm: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param decoder_fm:
        :param encoder_fm:
        :return:
        """
        decoder_fm = F.interpolate(decoder_fm, size=encoder_fm.size()[2:], mode="bilinear", align_corners=False)

        encoder_fm = self.skip(encoder_fm)
        x = decoder_fm + encoder_fm

        x = self.reduction(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.abn1(x)

        dsv = self.dsv(x)

        return x, dsv


class FPNSumDecoder(SegmentationDecoderModule):
    """

    """

    def __init__(
        self,
        feature_maps: List[int],
        num_classes: int,
        fpn_channels=256,
        dropout=0.0,
        abn_block=ABN,
        center_block=FPNSumCenterBlock,
        decoder_block=FPNSumDecoderBlock,
    ):
        super().__init__()

        self.center = center_block(
            feature_maps[-1], fpn_channels, num_classes=num_classes, dropout=dropout, abn_block=abn_block
        )

        self.fpn_modules = nn.ModuleList(
            [
                decoder_block(
                    encoder_fm, decoder_fm, decoder_fm, num_classes=num_classes, dropout=dropout, abn_block=abn_block
                )
                for decoder_fm, encoder_fm in zip(repeat(fpn_channels), reversed(feature_maps[:-1]))
            ]
        )

        self.final_block = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def forward(self, feature_maps: List[Tensor]) -> Tuple[Tensor, Tensor]:
        last_feature_map = feature_maps[-1]
        feature_maps = reversed(feature_maps[:-1])

        dsv_masks = []
        x, dsv = self.center(last_feature_map)

        dsv_masks.append(dsv)

        for transition_unit, encoder_fm in zip(self.fpn_modules, feature_maps):
            x, dsv = transition_unit(x, encoder_fm)
            dsv_masks.append(dsv)

        x = self.final_block(x)
        return x, dsv_masks
