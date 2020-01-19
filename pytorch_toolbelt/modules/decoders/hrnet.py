from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .common import SegmentationDecoderModule

__all__ = ["HRNetSegmentationDecoder"]


class HRNetSegmentationDecoder(SegmentationDecoderModule):
    def __init__(self, feature_maps: List[int], output_channels: int, dropout=0.0):
        super().__init__()

        features = sum(feature_maps)
        self.embedding = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
                    ),
                    ("bn1", nn.BatchNorm2d(features)),
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

    def forward(self, features: List[Tensor]):
        x_size = features[0].size()[2:]

        resized_feature_maps = [features[0]]
        for feature_map in features[1:]:
            feature_map = F.interpolate(feature_map, size=x_size, mode="nearest")
            resized_feature_maps.append(feature_map)

        feature_map = torch.cat(resized_feature_maps, dim=1)
        embedding = self.embedding(feature_map)
        return self.final(embedding)
