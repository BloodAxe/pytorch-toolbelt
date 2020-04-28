from typing import List, Union

import torch
from torch import nn, Tensor

from .common import SegmentationDecoderModule
from .. import conv1x1
from ..activations import ABN
from ..fpn import FPNContextBlock, FPNBottleneckBlock

__all__ = ["FPNCatDecoderBlock", "FPNCatDecoder"]


class FPNCatDecoderBlock(nn.Module):
    """
    Simple prediction block composed of (Conv + BN + Activation) repeated twice
    """

    def __init__(self, input_features: int, output_features: int, abn_block=ABN, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(output_features)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(output_features)
        self.drop2 = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.drop2(x)
        return x


class FPNCatDecoder(SegmentationDecoderModule):
    """
    Feature pyramid network decoder with concatenation between intermediate layers:

        Input
        fm[0] -> predict(concat(bottleneck[0](fm[0]), upsample(fpn[1]))) -> fpn[0] -> output[0](fpn[0])
        fm[1] -> predict(concat(bottleneck[1](fm[1]), upsample(fpn[2]))) -> fpn[1] -> output[1](fpn[1])
        fm[2] -> predict(concat(bottleneck[2](fm[2]), upsample(fpn[3]))) -> fpn[2] -> output[2](fpn[2])
        ...
        fm[n] -> predict(concat(bottleneck[n](fm[n]), upsample(context)) -> fpn[n] -> output[n](fpn[n])
        fm[n] -> context_block(feature_map[n]) -> context
    """

    def __init__(
        self,
        feature_maps: List[int],
        fpn_channels: int,
        context_block=FPNContextBlock,
        bottleneck_block=FPNBottleneckBlock,
        predict_block: Union[nn.Identity, conv1x1, nn.Module] = conv1x1,
        output_block: Union[nn.Identity, conv1x1, nn.Module] = nn.Identity,
        prediction_channels: int = None,
        upsample_block=nn.Upsample,
    ):
        """
        Create a new instance of FPN decoder with concatenation of consecutive feature maps.
        :param feature_maps: Number of channels in input feature maps (fine to coarse).
            For instance - [64, 256, 512, 2048]
        :param fpn_channels: FPN channels
        :param context_block:
        :param bottleneck_block:
        :param predict_block:
        :param output_block: Optional prediction block to apply to FPN feature maps before returning from decoder
        :param prediction_channels: Number of prediction channels
        :param upsample_block:
        """
        super().__init__()

        self.context = context_block(feature_maps[-1], fpn_channels)

        self.bottlenecks = nn.ModuleList(
            [bottleneck_block(in_channels, fpn_channels) for in_channels in reversed(feature_maps)]
        )

        self.predicts = nn.ModuleList(
            [predict_block(fpn_channels + fpn_channels, fpn_channels) for _ in reversed(feature_maps)]
        )

        if issubclass(output_block, nn.Identity):
            self.channels = [fpn_channels] * len(feature_maps)
            self.outputs = nn.ModuleList([output_block() for _ in reversed(feature_maps)])
        else:
            self.channels = [prediction_channels] * len(feature_maps)
            self.outputs = nn.ModuleList(
                [output_block(fpn_channels, prediction_channels) for _ in reversed(feature_maps)]
            )

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

        for feature_map, bottleneck, upsample, predict_block, output_block in zip(
            feature_maps, self.bottlenecks, self.upsamples, self.predicts, self.outputs
        ):
            fpn = torch.cat([bottleneck(feature_map), upsample(fpn)], dim=1)
            fpn = predict_block(fpn)
            outputs.append(output_block(fpn))

        # Returns list of tensors in same order as input (fine-to-coarse)
        return outputs[::-1]
