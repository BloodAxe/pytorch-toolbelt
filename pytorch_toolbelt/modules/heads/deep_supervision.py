from typing import List, Dict, Optional, Callable

import torch
import torch.nn as nn

from pytorch_toolbelt.datasets import name_for_stride
from pytorch_toolbelt.modules.simple import conv1x1

from pytorch_toolbelt.modules.interfaces import AbstractHead, FeatureMapsSpecification

__all__ = ["DeepSupervisionHead"]


class DeepSupervisionHead(AbstractHead):
    """
    Deep supervision head takes all feature maps and applies 1x1 convolution to each of them.
    """

    __constants__ = ["output_name_prefix", "output_spec"]

    def __init__(
        self, input_spec: FeatureMapsSpecification, num_classes: int, output_name_prefix: Optional[str] = None
    ):
        super().__init__(input_spec)
        self.heads = nn.ModuleList([conv1x1(channel, num_classes) for channel in input_spec.channels])
        self.output_spec = FeatureMapsSpecification(
            channels=tuple(
                [num_classes] * len(input_spec.channels),
            ),
            strides=input_spec.strides,
        )
        self.output_name_prefix = output_name_prefix

    def forward(self, feature_maps: List[torch.Tensor], output_size=None) -> Dict[str, torch.Tensor]:
        if self.output_name_prefix is None:
            outputs = []
            for feature_map, head, output_stride in zip(feature_maps, self.heads, self.output_spec.strides):
                output = head(feature_map)
                outputs.append(output)
        else:
            outputs = {}
            for feature_map, head, output_stride in zip(feature_maps, self.heads, self.output_spec.strides):
                output = head(feature_map)
                outputs[name_for_stride(self.output_name_prefix, output_stride)] = output

        return outputs

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        return self.output_spec

    def apply_to_final_layer(self, func: Callable[[nn.Module], None]):
        for head in self.heads:
            func(head)
