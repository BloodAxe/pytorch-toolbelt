import inspect
from typing import List, Union, Callable, Type

import torch
from torch import Tensor, nn

from pytorch_toolbelt.modules.interfaces import AbstractDecoder, FeatureMapsSpecification
from .. import conv1x1, conv3x3, instantiate_upsample_block, UpsampleLayerType, AbstractResizeLayer

__all__ = ["FPNDecoder"]


class FPNDecoder(AbstractDecoder):
    """
    Feature pyramid network decoder with summation between intermediate layers:

        Input
        feature_map[0] -> bottleneck[0](feature_map[0]) + upsample(fpn[1]) -> fpn[0]
        feature_map[1] -> bottleneck[1](feature_map[1]) + upsample(fpn[2]) -> fpn[1]
        feature_map[2] -> bottleneck[2](feature_map[2]) + upsample(fpn[3]) -> fpn[2]
        ...
        feature_map[n-1] -> bottleneck[n-1](feature_map[n-1]) + upsample(context) -> fpn[n]
        feature_map[n] -> context_block(feature_map[n]) -> context
    """

    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        out_channels: int,
        bottleneck_block: Callable[[int, int], nn.Module] = conv1x1,
        prediction_block: Union[nn.Identity, Callable[[int, int], nn.Module]] = conv3x3,
        upsample_block: Union[UpsampleLayerType, Type[AbstractResizeLayer]] = UpsampleLayerType.BILINEAR,
    ):
        """
        Create a new instance of FPN decoder with summation of consecutive feature maps.
        :param out_channels: Number of output channels in each feature map
        :param bottleneck_block:
        :param prediction_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__(input_spec)

        feature_maps = input_spec.channels

        self.lateral = nn.ModuleList([bottleneck_block(in_channels, out_channels) for in_channels in feature_maps])

        if inspect.isclass(prediction_block) and issubclass(prediction_block, nn.Identity):
            self.outputs = nn.ModuleList([prediction_block() for _ in feature_maps[:-1]])
        else:
            self.outputs = nn.ModuleList([prediction_block(out_channels, out_channels) for _ in feature_maps[:-1]])

        # TODO: Check that upsample_block we using produces the same number of output channels as input
        # TODO: Calculate the scale_factor form input_spec.strides
        num_upsample_blocks = len(feature_maps) - 1
        self.upsamples = nn.ModuleList(
            [
                instantiate_upsample_block(upsample_block, in_channels=out_channels, scale_factor=2)
                for _ in range(num_upsample_blocks)
            ]
        )

        self.output_spec = FeatureMapsSpecification(
            channels=[out_channels] * len(feature_maps), strides=input_spec.strides
        )

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        return self.output_spec

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        # Lateral connections
        lateral_maps = [lateral(feature_map) for feature_map, lateral in zip(feature_maps, self.lateral)]

        last_feature_map = lateral_maps[-1]
        remaining_feature_maps = lateral_maps[:-1][::-1]

        outputs = [last_feature_map]

        for feature_map, upsample, output_block in zip(remaining_feature_maps, self.upsamples, self.outputs):
            upsampled = upsample(outputs[-1], output_size=feature_map.shape[-2:])

            fpn = output_block(feature_map + upsampled)
            outputs.append(fpn)

        # Returns list of tensors in same order as input (fine-to-coarse)
        fine_to_coarse = outputs[::-1]
        return fine_to_coarse
