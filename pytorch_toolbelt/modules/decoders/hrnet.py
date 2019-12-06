from collections import OrderedDict

from torch import nn
from typing import List

from .common import DecoderModule
from ..backbone.hrnet import HRNETV2_BN_MOMENTUM

__all__ = ["HRNetDecoder"]


class HRNetDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], output_channels: int, dropout=0.0):
        super().__init__()

        features = feature_maps[-1]

        self.embedding = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
                    ),
                    ("bn1", nn.BatchNorm2d(features, momentum=HRNETV2_BN_MOMENTUM)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.logits = nn.Sequential(
            OrderedDict(
                [
                    ("drop", nn.Dropout2d(dropout)),
                    ("final", nn.Conv2d(in_channels=features, out_channels=output_channels, kernel_size=1)),
                ]
            )
        )

    def forward(self, features):
        embedding = self.embedding(features)
        return self.logits(embedding)
