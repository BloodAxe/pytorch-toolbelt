from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..activations import instantiate_activation_block, ACT_RELU
from ..interfaces import AbstractDecoder, FeatureMapsSpecification

__all__ = ["DeeplabV3PlusDecoder"]

from ..spp import ASPP, SeparableASPPModule


class DeeplabV3PlusDecoder(AbstractDecoder):
    """
    Implements DeepLabV3 model from `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Partially copy-pasted from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
    """

    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        out_channels: int,
        aspp_channels: int,
        low_level_channels: int = 48,
        atrous_rates=(12, 24, 36),
        dropout: float = 0.5,
        activation: str = ACT_RELU,
    ):
        """

        Args:
            feature_maps: Input feature maps
            aspp_channels:
            out_channels: Number of output channels
            atrous_rates:
            dropout:
            activation:
            low_level_channels:
        """
        super().__init__(input_spec)

        self.project = nn.Sequential(
            nn.Conv2d(input_spec.channels[0], low_level_channels, 1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            instantiate_activation_block(activation, inplace=True),
        )

        self.aspp = ASPP(
            in_channels=input_spec.channels[-1],
            out_channels=aspp_channels,
            atrous_rates=atrous_rates,
            dropout=dropout,
            activation=activation,
            aspp_module=SeparableASPPModule,
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                aspp_channels + low_level_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            instantiate_activation_block(activation, inplace=True),
        )

        self.output_spec = FeatureMapsSpecification(
            channels=[out_channels, aspp_channels], strides=[input_spec.strides[0], input_spec.strides[-1]]
        )

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        coarse_features = self.aspp(feature_maps[-1])
        low_level_features = self.project(feature_maps[0])

        coarse_features_resized = F.interpolate(
            coarse_features, size=low_level_features.shape[2:], mode="bilinear", align_corners=False
        )
        combined_features = torch.cat([low_level_features, coarse_features_resized], dim=1)
        return [self.final(combined_features), coarse_features]

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        return self.output_spec
