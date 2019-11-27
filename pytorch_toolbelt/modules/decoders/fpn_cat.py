from typing import List, Tuple

from torch import nn, Tensor

from .common import SegmentationDecoderModule
from .fpn import FPNDecoder
from ..activated_batch_norm import ABN
from ..fpn import FPNFuse, UpsampleAdd

__all__ = ["FPNCatDecoder"]


class FPNCatDecoderBlock(nn.Module):
    """
    Simple prediction block composed of (Conv + BN + Activation) repeated twice
    """

    def __init__(self, input_features: int, output_features: int, abn_block=ABN, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(output_features)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(output_features)
        self.drop2 = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.drop2(x)
        return x


class FPNCatDecoder(SegmentationDecoderModule):
    """

    """

    def __init__(
        self,
        feature_maps: List[int],
        num_classes: int,
        fpn_channels=128,
        dropout=0.0,
        abn_block=ABN,
        upsample_add=UpsampleAdd,
        prediction_block=FPNCatDecoderBlock,
    ):
        super().__init__()

        self.fpn = FPNDecoder(
            feature_maps,
            upsample_add_block=upsample_add,
            prediction_block=prediction_block,
            fpn_features=fpn_channels,
            prediction_features=fpn_channels,
        )

        self.fuse = FPNFuse()
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        # dsv blocks are for deep supervision
        self.dsv = nn.ModuleList(
            [
                nn.Conv2d(fpn_features, num_classes, kernel_size=1)
                for fpn_features in [fpn_channels] * len(feature_maps)
            ]
        )

        features = sum(self.fpn.output_filters)

        self.final_block = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=1),
            abn_block(features // 2),
            nn.Conv2d(features // 2, features // 4, kernel_size=3, padding=1, bias=False),
            abn_block(features // 4),
            nn.Conv2d(features // 4, features // 4, kernel_size=3, padding=1, bias=False),
            abn_block(features // 4),
            nn.Conv2d(features // 4, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, feature_maps: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        fpn_maps = self.fpn(feature_maps)

        fused = self.fuse(fpn_maps)
        fused = self.dropout(fused)

        dsv_masks = []
        for dsv_block, fpn in zip(self.dsv, fpn_maps):
            dsv = dsv_block(fpn)
            dsv_masks.append(dsv)

        x = self.final_block(fused)
        return x, dsv_masks
