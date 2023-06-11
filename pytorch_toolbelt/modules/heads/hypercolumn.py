from typing import Optional, List

import torch.jit
from torch import nn, Tensor

from pytorch_toolbelt.modules import FPNFuse, instantiate_activation_block, instantiate_normalization_block
from pytorch_toolbelt.modules.interfaces import AbstractHead, FeatureMapsSpecification

__all__ = ["HypercolumnHead"]


class HypercolumnHead(AbstractHead):
    """
    Hypercolumn head that concatenates all input feature maps to the size of the largest feature map, compute
    a projection (using conv + bn + act) and resize the projected feature map to the original image size.

    Reference: https://arxiv.org/pdf/1411.5752.pdf
    """

    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        activation: str,
        normalization: str,
        mid_channels: int,
        output_name: Optional[str] = None,
        dropout_rate: float = 0.0,
        dropout_inplace: bool = False,
        interpolation_mode="bilinear",
        interpolation_align_corners=False,
    ):
        super().__init__(input_spec)
        channels = sum(input_spec.channels)
        self.fuse = FPNFuse(mode=interpolation_mode, align_corners=interpolation_align_corners)

        self.projection = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            instantiate_normalization_block(normalization, mid_channels),
            instantiate_activation_block(activation, inplace=True),
            nn.Dropout2d(dropout_rate, inplace=dropout_inplace),
        )

        self.final = nn.Conv2d(mid_channels, num_classes, kernel_size=3, padding=1)
        self.output_name = output_name
        self.output_spec = FeatureMapsSpecification(channels=(num_classes,), strides=(1,))
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners

    def forward(self, feature_maps: List[Tensor], output_size: torch.Size):
        x = self.fuse(feature_maps)
        x = self.projection(x)
        x = self.final(x)

        output = torch.nn.functional.interpolate(
            x, size=output_size, mode=self.interpolation_mode, align_corners=self.interpolation_align_corners
        )

        if self.output_name is not None:
            return {self.output_name: output}
        else:
            return output
