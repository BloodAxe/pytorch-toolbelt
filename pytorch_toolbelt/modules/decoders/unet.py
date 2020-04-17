from typing import List, Union

import torch
from torch import nn

from .common import DecoderModule
from ..simple import conv1x1
from ..activations import ABN
from ..unet import UnetBlock, UnetDecoderBlock

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
        for block_index in reversed(range(num_blocks)):
            features_from_encoder = feature_maps[block_index]

            if isinstance(upsample_block, nn.Upsample):
                upsamples.append(upsample_block)
                features_from_upsample = feature_maps[block_index + 1]
            elif issubclass(upsample_block, nn.Upsample):
                upsamples.append(upsample_block(scale_factor=2))
                features_from_upsample = feature_maps[block_index + 1]
            elif issubclass(upsample_block, nn.ConvTranspose2d):
                up = upsample_block(
                    feature_maps[block_index + 1], feature_maps[block_index + 1] // 2, kernel_size=3, stride=2,
                    padding=1,
                )
                upsamples.append(up)
                features_from_upsample = up.out_channels
            else:
                up = upsample_block(feature_maps[block_index + 1])
                upsamples.append(up)
                features_from_upsample = up.out_channels

            in_channels = features_from_encoder + features_from_upsample
            out_channels = in_channels // 2
            blocks.append(unet_block(in_channels, out_channels))

        self.blocks = nn.ModuleList(blocks)
        self.upsamples = nn.ModuleList(upsamples)
        self.output_filters = decoder_features

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        x = feature_maps[-1]
        outputs = []
        num_feature_maps = len(feature_maps)
        for index, (upsample_block, decoder_block) in enumerate(zip(self.upsamples, self.blocks)):
            encoder_input = feature_maps[num_feature_maps - index - 2]

            if isinstance(upsample_block, nn.ConvTranspose2d):
                x = upsample_block(x, output_size=encoder_input.size())

            x = torch.cat([x, encoder_input], dim=1)
            x = decoder_block(x)
            outputs.append(x)

        # Returns list of tensors in same order as input (fine-to-coarse)
        return outputs[::-1]
