from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from ..activated_batch_norm import ABN
from .common import DecoderModule
from ..unet import UnetCentralBlock, UnetDecoderBlock

__all__ = ["UNetDecoder"]


def conv1x1(input, output):
    return nn.Conv2d(input, output, kernel_size=1)


class UNetDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], decoder_features: int, mask_channels: int, abn_block=ABN, dropout=0.0, final_block=conv1x1):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]
        else:
            assert len(decoder_features) == len(feature_maps)

        self.center = UnetCentralBlock(
            in_dec_filters=feature_maps[-1], out_filters=decoder_features[-1], abn_block=abn_block
        )

        blocks = []
        for block_index, in_enc_features in enumerate(feature_maps[:-1]):
            blocks.append(
                UnetDecoderBlock(
                    in_dec_filters=decoder_features[block_index + 1],
                    in_enc_filters=in_enc_features,
                    out_filters=decoder_features[block_index],
                    abn_block=abn_block,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.output_filters = decoder_features

        self.final_drop = nn.Dropout2d(dropout)
        self.final = final_block(decoder_features[0], mask_channels)

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        output = self.center(feature_maps[-1])

        for decoder_block, encoder_output in zip(reversed(self.blocks), reversed(feature_maps[:-1])):
            x = decoder_block(output, encoder_output)
            output = x

        output = self.final_drop(output)
        output = self.final(output)
        return output
