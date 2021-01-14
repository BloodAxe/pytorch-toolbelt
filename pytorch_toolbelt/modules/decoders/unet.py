from typing import List, Union

import torch
from torch import nn

from .common import DecoderModule
from .. import DeconvolutionUpsample2d
from ..unet import UnetBlock

__all__ = ["UNetDecoder"]


class UNetDecoder(DecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: Union[int, List[int]] = None,
        unet_block=UnetBlock,
        upsample_block: Union[nn.Upsample, nn.ConvTranspose2d] = None,
    ):
        super().__init__()

        # if not isinstance(decoder_features, list):
        #     decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]
        # else:
        #     assert len(decoder_features) == len(
        #         feature_maps
        #     ), f"Incorrect number of decoder features: {decoder_features}, {feature_maps}"

        if upsample_block is None:
            upsample_block = nn.ConvTranspose2d

        blocks = []
        upsamples = []

        num_blocks = len(feature_maps) - 1  # Number of outputs is one less than encoder layers

        if decoder_features is None:
            decoder_features = [None] * num_blocks
        else:
            if len(decoder_features) != num_blocks:
                raise ValueError(f"decoder_features must have length of {num_blocks}")
        in_channels_for_upsample_block = feature_maps[-1]

        for block_index in reversed(range(num_blocks)):
            features_from_encoder = feature_maps[block_index]

            if isinstance(upsample_block, nn.Upsample):
                upsamples.append(upsample_block)
                out_channels_from_upsample_block = in_channels_for_upsample_block
            elif issubclass(upsample_block, nn.Upsample):
                upsamples.append(upsample_block(scale_factor=2))
                out_channels_from_upsample_block = in_channels_for_upsample_block
            elif issubclass(upsample_block, nn.ConvTranspose2d):
                up = upsample_block(
                    in_channels_for_upsample_block,
                    in_channels_for_upsample_block // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                upsamples.append(up)
                out_channels_from_upsample_block = up.out_channels
            else:
                up = upsample_block(in_channels_for_upsample_block)
                upsamples.append(up)
                out_channels_from_upsample_block = up.out_channels

            in_channels = features_from_encoder + out_channels_from_upsample_block
            out_channels = decoder_features[block_index] or in_channels // 2
            blocks.append(unet_block(in_channels, out_channels))

            in_channels_for_upsample_block = out_channels
            decoder_features[block_index] = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.output_filters = decoder_features

    @property
    def channels(self) -> List[int]:
        return self.output_filters

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, (upsample_block, decoder_block) in enumerate(zip(self.upsamples, self.blocks)):
            encoder_input = feature_maps[num_feature_maps - index - 2]

            if isinstance(upsample_block, (nn.ConvTranspose2d, DeconvolutionUpsample2d)):
                x = upsample_block(x, output_size=encoder_input.size())
            else:
                x = upsample_block(x)

            x = torch.cat([x, encoder_input], dim=1)
            x = decoder_block(x)
            outputs.append(x)

        # Returns list of tensors in same order as input (fine-to-coarse)
        return outputs[::-1]
