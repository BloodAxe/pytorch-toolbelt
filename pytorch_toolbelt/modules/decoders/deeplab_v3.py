from typing import List

import torch
from torch import nn, Tensor

__all__ = ["DeeplabV3Decoder"]

from ..activations import instantiate_activation_block, ACT_RELU
from ..interfaces import AbstractDecoder, FeatureMapsSpecification
from ..spp import ASPPModule, ASPP


class DeeplabV3Decoder(AbstractDecoder):
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
        atrous_rates=(12, 24, 36),
        dropout: float = 0.5,
        activation=ACT_RELU,
    ):
        """

        Args:
            feature_maps: List of input channels
            aspp_channels:
            out_channels: Output channels
            atrous_rates:
            dropout:
            activation:
        """
        super().__init__(input_spec)
        self.aspp = ASPP(
            in_channels=input_spec.channels[-1],
            out_channels=aspp_channels,
            aspp_module=ASPPModule,
            atrous_rates=atrous_rates,
            dropout=dropout,
            activation=activation,
        )
        self.final = nn.Sequential(
            nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(aspp_channels, out_channels, kernel_size=1),
        )

        self.output_spec = FeatureMapsSpecification(channels=[out_channels], strides=[input_spec.strides[-1]])

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        high_level_features = feature_maps[-1]
        high_level_features = self.aspp(high_level_features)
        return self.final(high_level_features)

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        return self.output_spec
