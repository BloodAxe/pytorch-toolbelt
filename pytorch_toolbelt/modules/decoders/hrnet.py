from torch import nn

from .common import DecoderModule
from ..backbone.hrnet import HRNETV2_BN_MOMENTUM

__all__ = ["HRNetDecoder"]


class HRNetDecoder(DecoderModule):
    def __init__(self, features: int, num_classes: int, dropout=0.0):
        super().__init__()

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(features, momentum=HRNETV2_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=features, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, features):
        return self.last_layer(features[-1])
