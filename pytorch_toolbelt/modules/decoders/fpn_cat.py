from typing import List, Tuple

import torch
from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules.decoders import DecoderModule
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn, Tensor

from torch.nn import functional as F
from pytorch_toolbelt.modules.fpn import FPNFuse, UpsampleAdd

__all__ = ["FPNCatDecoder"]


class FPNCatDecoder(DecoderModule):
    """

    """

    def __init__(
        self,
        feature_maps: List[int],
        num_classes: int,
        fpn_channels=128,
        dropout=0.0,
        abn_block=ABN,
    ):
        super().__init__()

        self.fpn = FPNDecoder(
            feature_maps,
            upsample_add_block=UpsampleAdd,
            prediction_block=DoubleConvBNRelu,
            fpn_features=fpn_channels,
            prediction_features=fpn_channels,
        )

        self.fuse = FPNFuse()
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        self.dsv = nn.ModuleList(
            [
                nn.Conv2d(fpn_features, num_classes, kernel_size=1)
                for fpn_features in [fpn_channels] * len(feature_maps)
            ]
        )

        features = sum(self.fpn.output_filters)

        self.final_block = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=1),
            nn.BatchNorm2d(features // 2),
            nn.Conv2d(
                features // 2, features // 4, kernel_size=3, padding=1, bias=True
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(features // 4),
            nn.Conv2d(
                features // 4, features // 4, kernel_size=3, padding=1, bias=False
            ),
            nn.LeakyReLU(inplace=True),
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


@torch.no_grad()
def test_fpn_cat():
    channels = [256, 512, 1024, 2048]
    sizes = [64, 32, 16, 8]

    net = FPNCatDecoder(channels, 5).eval()

    input = [torch.randn(4, ch, sz, sz) for sz, ch in zip(sizes, channels)]
    output, dsv_masks = net(input)

    print(output.size(), output.mean(), output.std())
    for dsv in dsv_masks:
        print(dsv.size(), dsv.mean(), dsv.std())
    print(count_parameters(net))
