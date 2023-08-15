from typing import List, Union, Type, Tuple

import torch
import logging

from torch import nn

from pytorch_toolbelt.modules.unet import UnetBlock, UnetResidualBlock
from pytorch_toolbelt.modules.interfaces import AbstractDecoder, FeatureMapsSpecification
from pytorch_toolbelt.modules.upsample import (
    AbstractResizeLayer,
    UpsampleLayerType,
    instantiate_upsample_block,
    DeconvolutionUpsample2d,
)
from pytorch_toolbelt.modules.normalization import NORM_BATCH, instantiate_normalization_block
from pytorch_toolbelt.modules.activations import ACT_RELU

__all__ = ["UNetDecoder"]

logger = logging.getLogger(__name__)


class UNetDecoder(AbstractDecoder):
    def __init__(
        self,
        input_spec: FeatureMapsSpecification,
        out_channels: Union[Tuple[int, ...], List[int]],
        block_type: Union[Type[UnetBlock], Type[UnetResidualBlock]] = UnetBlock,
        upsample_block: Union[UpsampleLayerType, Type[AbstractResizeLayer]] = UpsampleLayerType.BILINEAR,
        activation: str = ACT_RELU,
        normalization: str = NORM_BATCH,
        block_kwargs=None,
        unet_block=None,
    ):
        if unet_block is not None:
            logger.warning("unet_block argument is deprecated, use block_type instead", DeprecationWarning)
            block_type = unet_block

        super().__init__(input_spec)
        if block_kwargs is None:
            block_kwargs = {
                "activation": activation,
                "normalization": normalization,
            }

        blocks = []
        upsamples = []

        num_blocks = len(input_spec) - 1  # Number of outputs is one less than encoder layers

        if len(out_channels) != num_blocks:
            raise ValueError(f"decoder_features must have length of {num_blocks}")

        in_channels_for_upsample_block = input_spec.channels[-1]

        for block_index in reversed(range(num_blocks)):
            features_from_encoder = input_spec.channels[block_index]

            scale_factor = input_spec.strides[block_index + 1] // input_spec.strides[block_index]
            upsample_layer: AbstractResizeLayer = instantiate_upsample_block(
                upsample_block, in_channels=in_channels_for_upsample_block, scale_factor=scale_factor
            )

            upsamples.append(upsample_layer)
            out_channels_from_upsample_block = upsample_layer.out_channels

            in_channels = features_from_encoder + out_channels_from_upsample_block

            blocks.append(block_type(in_channels, out_channels[block_index], **block_kwargs))

            in_channels_for_upsample_block = out_channels[block_index]

        self.blocks = nn.ModuleList(blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.output_spec = FeatureMapsSpecification(channels=out_channels, strides=input_spec.strides[:-1])

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        return self.output_spec

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, (upsample_block, decoder_block) in enumerate(zip(self.upsamples, self.blocks)):
            encoder_input = feature_maps[num_feature_maps - index - 2]

            x = upsample_block(x, output_size=encoder_input.size()[2:])

            x = torch.cat([x, encoder_input], dim=1)
            x = decoder_block(x)
            outputs.append(x)

        # Returns list of tensors in same order as input (fine-to-coarse)
        return outputs[::-1]
