from typing import List, Union
from torch import Tensor, nn
import inspect

from .common import SegmentationDecoderModule
from .. import conv1x1, FPNContextBlock, FPNBottleneckBlock

__all__ = ["FPNSumDecoder"]


class FPNSumDecoder(SegmentationDecoderModule):
    """
    Feature pyramid network decoder with summation between intermediate layers:

        Input
        feature_map[0] -> bottleneck[0](feature_map[0]) + upsample(fpn[1]) -> fpn[0]
        feature_map[1] -> bottleneck[1](feature_map[1]) + upsample(fpn[2]) -> fpn[1]
        feature_map[2] -> bottleneck[2](feature_map[2]) + upsample(fpn[3]) -> fpn[2]
        ...
        feature_map[n] -> bottleneck[n](feature_map[n]) + upsample(context) -> fpn[n]
        feature_map[n] -> context_block(feature_map[n]) -> context
    """

    def __init__(
        self,
        feature_maps: List[int],
        fpn_channels: int,
        context_block=FPNContextBlock,
        bottleneck_block=FPNBottleneckBlock,
        prediction_block: Union[nn.Identity, conv1x1, nn.Module] = nn.Identity,
        prediction_channels: int = None,
        upsample_block=nn.Upsample,
    ):
        """
        Create a new instance of FPN decoder with summation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param fpn_channels: FPN channels
        :param context_block:
        :param bottleneck_block:
        :param prediction_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__()

        self.context = context_block(feature_maps[-1], fpn_channels)

        self.bottlenecks = nn.ModuleList(
            [bottleneck_block(in_channels, fpn_channels) for in_channels in reversed(feature_maps)]
        )

        if inspect.isclass(prediction_block) and issubclass(prediction_block, nn.Identity):
            self.outputs = nn.ModuleList([prediction_block() for _ in reversed(feature_maps)])
            self.channels = [fpn_channels] * len(feature_maps)
        else:
            self.outputs = nn.ModuleList(
                [prediction_block(fpn_channels, prediction_channels) for _ in reversed(feature_maps)]
            )
            self.channels = [prediction_channels] * len(feature_maps)

        if issubclass(upsample_block, nn.Upsample):
            self.upsamples = nn.ModuleList([upsample_block(scale_factor=2) for _ in reversed(feature_maps)])
        else:
            self.upsamples = nn.ModuleList(
                [upsample_block(fpn_channels, fpn_channels) for in_channels in reversed(feature_maps)]
            )

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        last_feature_map = feature_maps[-1]
        feature_maps = reversed(feature_maps)

        outputs = []

        fpn = self.context(last_feature_map)

        for feature_map, bottleneck, upsample, output_block in zip(
            feature_maps, self.bottlenecks, self.upsamples, self.outputs
        ):
            fpn = bottleneck(feature_map) + upsample(fpn)
            outputs.append(output_block(fpn))

        # Returns list of tensors in same order as input (fine-to-coarse)
        return outputs[::-1]
