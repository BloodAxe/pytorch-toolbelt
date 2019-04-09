import torch
from torch import nn

from .unet import UnetCentralBlock, UnetDecoderBlock
from .fpn import FPNDecoderBlock, FPNBlock, FPNFuse

import torch.nn.functional as F


class DecoderModule(nn.Module):
    def __init__(self, features, strides):
        super().__init__()

    def forward(self, features):
        raise NotImplementedError


class UNetDecoder(DecoderModule):
    def __init__(self, features, strides, start_features: int, dilation_factors=[1, 1, 1, 1], **kwargs):
        super().__init__(features, strides)
        decoder_features = start_features
        reversed_features = list(reversed(features))

        output_filters = [decoder_features]
        self.center = UnetCentralBlock(reversed_features[0], decoder_features)

        if dilation_factors is None:
            dilation_factors = [1] * len(reversed_features)

        blocks = []
        for block_index, encoder_features in enumerate(reversed_features):
            blocks.append(UnetDecoderBlock(output_filters[-1], encoder_features, decoder_features, dilation=dilation_factors[block_index]))
            output_filters.append(decoder_features)
            # print(block_index, decoder_features, encoder_features, decoder_features)
            decoder_features = decoder_features // 2

        self.blocks = nn.ModuleList(blocks)
        self.output_filters = output_filters

    def forward(self, features):
        reversed_features = list(reversed(features))
        decoder_outputs = [self.center(reversed_features[0])]

        for block_index, decoder_block, encoder_output in zip(range(len(self.blocks)), self.blocks, reversed_features):
            # print(block_index, decoder_outputs[-1].size(), encoder_output.size())
            decoder_outputs.append(decoder_block(decoder_outputs[-1], encoder_output))

        return decoder_outputs


class FPNDecoder(DecoderModule):
    def __init__(self, features, strides, decoder_features=256, fpn_features=128, dropout=0.0, dilation=None, **kwargs):
        super().__init__(features, strides)
        reversed_features = list(reversed(features))

        decoder_blocks = []
        fpn_blocks = []
        if dilation is None:
            dilation = [1] * len(features)

        for block_index, (encoder_features, decoder_dilation) in enumerate(zip(reversed_features, reversed(dilation))):
            decoder_blocks.append(FPNDecoderBlock(encoder_features, decoder_features))
            fpn_blocks.append(FPNBlock(decoder_features, fpn_features, dropout=dropout, dilation=decoder_dilation))

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.fpn_blocks = nn.ModuleList(fpn_blocks)
        self.fpn_fuse = FPNFuse()
        self.output_filters = [len(fpn_blocks) * fpn_features]

    def forward(self, features):
        reversed_features = list(reversed(features))
        decoder_outputs = []
        fpn_outputs = []
        for block_index, decoder_block, fpn_block, encoder_output in zip(range(len(self.decoder_blocks)), self.decoder_blocks, self.fpn_blocks, reversed_features):
            decoder = decoder_block(encoder_output, decoder_outputs[-1] if len(decoder_outputs) else None)
            decoder_outputs.append(decoder)

            fpn = fpn_block(decoder)
            fpn_outputs.append(fpn)

        x = self.fpn_fuse(fpn_outputs)
        return [x]
