from torch import nn
from typing import List

from .common import DecoderModule
from ..backbone.hrnet import HRNETV2_BN_MOMENTUM

__all__ = ["HRNetDecoder"]


class HRNetDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], output_channels: int, dropout=0.0):
        super().__init__()

        features = feature_maps[-1]

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features, momentum=HRNETV2_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels=features, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, features):
        return self.last_layer(features[-1])
