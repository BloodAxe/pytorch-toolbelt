import torch
from torch import nn

from .unet import UnetCentralBlock, UnetDecoderBlock
from .fpn import FPNFuse, FPNBottleneckBlock, FPNPredictionBlock

import torch.nn.functional as F


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        raise NotImplementedError


class UNetDecoder(DecoderModule):
    def __init__(self, features, start_features: int, dilation_factors=[1, 1, 1, 1], **kwargs):
        super().__init__()
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
    def __init__(self, features,
                 fpn_features=128,
                 bottleneck=FPNBottleneckBlock,
                 prediction=FPNPredictionBlock,
                 **kwargs):
        super().__init__()

        self.bottlenecks = nn.ModuleList([bottleneck(input_channels, fpn_features) for input_channels in features])
        self.predictors = nn.ModuleList([prediction(fpn_features, fpn_features) for _ in features])

        self.output_filters = [fpn_features] * len(features)

    def forward(self, features):
        fpn_outputs = []

        for feature_map, bottleneck, fpn_block in zip(reversed(features),
                                                      reversed(self.bottlenecks),
                                                      reversed(self.predictors)):
            feature_map = bottleneck(feature_map)
            prev_fpn = fpn_outputs[-1] if len(fpn_outputs) else None
            y = fpn_block(feature_map, prev_fpn)
            fpn_outputs.append(y)

        # Reverse list of fpn features to match with order of input features
        return list(reversed(fpn_outputs))
