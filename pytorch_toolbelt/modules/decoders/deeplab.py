from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .common import DecoderModule
from ..activations import ABN

__all__ = ["DeeplabV3Decoder"]


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, abn_block=ABN):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.abn = abn_block(planes)

    def forward(self, x):  # skipcq: PYL-W0221
        x = self.atrous_conv(x)
        x = self.abn(x)
        return x


class ASPP(nn.Module):
    def __init__(self, inplanes: int, output_stride: int, output_features: int, dropout=0.5, abn_block=ABN):
        super(ASPP, self).__init__()

        if output_stride == 32:
            dilations = [1, 3, 6, 9]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, output_features, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, output_features, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, output_features, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, output_features, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, output_features, 1, stride=1, bias=False),
            abn_block(output_features),
        )
        self.conv1 = nn.Conv2d(output_features * 5, output_features, 1, bias=False)
        self.abn1 = abn_block(output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # skipcq: PYL-W0221
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.abn1(x)

        return self.dropout(x)


class DeeplabV3Decoder(DecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        num_classes: int,
        output_stride=32,
        high_level_bottleneck=256,
        low_level_bottleneck=32,
        dropout=0.5,
        abn_block=ABN,
    ):
        super(DeeplabV3Decoder, self).__init__()

        self.aspp = ASPP(feature_maps[-1], output_stride, high_level_bottleneck, dropout=dropout, abn_block=abn_block)

        self.conv1 = nn.Conv2d(feature_maps[0], low_level_bottleneck, 1, bias=False)
        self.abn1 = abn_block(low_level_bottleneck)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                high_level_bottleneck + low_level_bottleneck,
                high_level_bottleneck,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            abn_block(high_level_bottleneck),
            nn.Dropout(dropout),
            nn.Conv2d(high_level_bottleneck, high_level_bottleneck, kernel_size=3, padding=1, bias=False),
            abn_block(high_level_bottleneck),
            nn.Dropout(dropout * 0.2),  # 5 times smaller dropout rate
            nn.Conv2d(high_level_bottleneck, num_classes, kernel_size=1),
        )

        self.dsv = nn.Conv2d(high_level_bottleneck, num_classes, kernel_size=1)

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        low_level_feat = feature_maps[0]
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.abn1(low_level_feat)

        high_level_features = feature_maps[-1]
        high_level_features = self.aspp(high_level_features)

        mask_dsv = self.dsv(high_level_features)

        high_level_features = F.interpolate(
            high_level_features, size=low_level_feat.size()[2:], mode="bilinear", align_corners=False
        )
        high_level_features = torch.cat([high_level_features, low_level_feat], dim=1)
        mask = self.last_conv(high_level_features)

        return [mask, mask_dsv]
