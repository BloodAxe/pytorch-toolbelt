from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_toolbelt.datasets import name_for_stride
from pytorch_toolbelt.modules import instantiate_activation_block, ACT_GELU
from pytorch_toolbelt.modules.interfaces import AbstractHead, FeatureMapsSpecification


__all__ = ["SegFormerHead"]


class SegFormerHead(AbstractHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        num_classes: int,
        embedding_dim: int,
        with_supervision: bool,
        output_name: Optional[str],
        dropout_rate: float = 0.0,
        activation: str = ACT_GELU,
    ):
        super().__init__(input_spec)

        self.num_classes = num_classes
        self.output_name = output_name
        self.with_supervision = with_supervision

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = input_spec.channels

        self.linear_c4 = nn.Conv2d(c4_in_channels, embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(c3_in_channels, embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(c2_in_channels, embedding_dim, kernel_size=1)
        self.linear_c1 = nn.Conv2d(c1_in_channels, embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            instantiate_activation_block(activation, inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_rate, inplace=False)
        self.final = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

        if self.with_supervision:
            self.supervision_c4 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
            self.supervision_c3 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
            self.supervision_c2 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
            self.supervision_c1 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        else:
            self.supervision_c4 = None
            self.supervision_c3 = None
            self.supervision_c2 = None
            self.supervision_c1 = None

    def forward(self, feature_maps: List[Tensor], output_size):
        c1, c2, c3, c4 = feature_maps

        c4 = self.linear_c4(c4)
        c3 = self.linear_c3(c3)
        c2 = self.linear_c2(c2)
        c1 = self.linear_c1(c1)

        c = self.linear_fuse(
            torch.cat(
                [
                    torch.nn.functional.interpolate(c4, size=c1.size()[2:], mode="bilinear", align_corners=False),
                    torch.nn.functional.interpolate(c3, size=c1.size()[2:], mode="bilinear", align_corners=False),
                    torch.nn.functional.interpolate(c2, size=c1.size()[2:], mode="bilinear", align_corners=False),
                    c1,
                ],
                dim=1,
            )
        )

        x = self.dropout(c)
        x = self.final(x)
        x = torch.nn.functional.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        if self.output_name is not None:
            outputs = {self.output_name: x}
        else:
            outputs = x

        if self.with_supervision:
            s4, s3, s2, s1 = (
                self.supervision_c4(c4),
                self.supervision_c3(c3),
                self.supervision_c2(c2),
                self.supervision_c1(c1),
            )

            if self.output_name is not None:
                outputs[name_for_stride(self.output_name, 32)] = s4
                outputs[name_for_stride(self.output_name, 16)] = s3
                outputs[name_for_stride(self.output_name, 8)] = s2
                outputs[name_for_stride(self.output_name, 4)] = s1
            else:
                outputs = outputs + (s1, s2, s3, s4)

        return outputs

    def apply_to_final_layer(self, func):
        func(self.final)
        if self.with_supervision:
            func(self.supervision_c4)
            func(self.supervision_c3)
            func(self.supervision_c2)
            func(self.supervision_c1)
