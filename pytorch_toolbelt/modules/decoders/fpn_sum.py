from typing import List, Union, Callable, Tuple

import torch
from torch import Tensor, nn
import inspect

from .common import SegmentationDecoderModule
from .. import conv1x1, conv3x3, FPNContextBlock, FPNBottleneckBlock, InterpolateLayer

__all__ = ["FPNSumDecoder"]


class FPNSumDecoder(SegmentationDecoderModule):
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
        feature_maps: List[int],
        strides: List[int],
        channels: int,
        bottleneck_block: Callable[[int, int], nn.Module] = conv1x1,
        prediction_block: Union[nn.Identity, Callable[[int, int], nn.Module]] = conv3x3,
        upsample_type: Union[str, nn.Module] = "interpolate",
        upsample_kwargs=None,
    ):
        """
        Create a new instance of FPN decoder with summation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param channels: FPN channels
        :param bottleneck_block:
        :param prediction_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        :param upsample_kwargs:
        """
        super().__init__()
        if upsample_kwargs is None:
            upsample_kwargs = {}
        self.upsample_kwargs = upsample_kwargs

        self.lateral = nn.ModuleList([bottleneck_block(in_channels, channels) for in_channels in feature_maps])

        if inspect.isclass(prediction_block) and issubclass(prediction_block, nn.Identity):
            self.outputs = nn.ModuleList([prediction_block() for _ in feature_maps[:-1]])
        else:
            self.outputs = nn.ModuleList([prediction_block(channels, channels) for _ in feature_maps[:-1]])

        if inspect.isclass(upsample_type) and issubclass(upsample_type, nn.Upsample):
            self.upsamples = nn.ModuleList([upsample_type(scale_factor=2) for _ in reversed(feature_maps[:-1])])
        elif isinstance(upsample_type,str) and upsample_type == "interpolate":
            self.upsamples = nn.ModuleList(
                [InterpolateLayer(self.upsample_kwargs) for _ in reversed(feature_maps[:-1])]
            )
        else:
            raise ValueError(f"Unknown upsample type: {upsample_type}")

        self._channels = tuple([channels] * len(feature_maps))
        self._strides = tuple(strides)

    @property
    @torch.jit.ignore
    def channels(self):
        return self._channels

    @property
    @torch.jit.ignore
    def strides(self):
        return self._strides

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        # Lateral connections
        lateral_maps = [lateral(feature_map) for feature_map, lateral in zip(feature_maps, self.lateral)]

        last_feature_map = lateral_maps[-1]
        remaining_feature_maps = lateral_maps[:-1][::-1]

        outputs = [last_feature_map]

        for feature_map, upsample, output_block in zip(remaining_feature_maps, self.upsamples, self.outputs):
            if isinstance(upsample, (InterpolateLayer, nn.ConvTranspose2d)):
                upsampled = upsample(outputs[-1], output_size=feature_map.shape[-2:])
            else:
                upsampled = upsample(outputs[-1])

            fpn = output_block(feature_map + upsampled)
            outputs.append(fpn)

        # Returns list of tensors in same order as input (fine-to-coarse)
        fine_to_coarse = outputs[::-1]
        return fine_to_coarse
