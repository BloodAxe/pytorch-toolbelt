from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from ..activated_batch_norm import ABN
from .common import DecoderModule
from ..unet import UnetCentralBlock, UnetDecoderBlock

__all__ = ["UNetDecoder"]


class UNetDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], decoder_features: int, mask_channels: int):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]

        blocks = []
        for block_index, in_enc_features in enumerate(feature_maps[:-1]):
            blocks.append(
                UnetDecoderBlock(
                    decoder_features[block_index + 1], in_enc_features, decoder_features[block_index], mask_channels
                )
            )

        self.center = UnetCentralBlock(feature_maps[-1], decoder_features[-1], mask_channels)
        self.blocks = nn.ModuleList(blocks)
        self.output_filters = decoder_features

    def forward(self, feature_maps):

        output, dsv = self.center(feature_maps[-1])
        decoder_outputs = [output]
        dsv_list = [dsv]

        for decoder_block, encoder_output in zip(reversed(self.blocks), reversed(feature_maps[:-1])):
            output, dsv = decoder_block(output, encoder_output)
            decoder_outputs.append(output)
            dsv_list.append(dsv)

        dsv_list = list(reversed(dsv_list))
        decoder_outputs = list(reversed(decoder_outputs))

        return decoder_outputs, dsv_list
