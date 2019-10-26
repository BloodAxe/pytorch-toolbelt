from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from ..abn import ABN
from .common import DecoderModule

__all__ = ["UnetCentralBlock", "UnetDecoderBlock", "UNetDecoder"]


class UnetCentralBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, abn_block=ABN, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_dec_filters,
            out_filters,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False,
            **kwargs
        )
        self.bn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(
            out_filters, out_filters, kernel_size=3, padding=1, bias=False, **kwargs
        )
        self.bn2 = abn_block(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UnetDecoderBlock(nn.Module):
    """
    """

    def __init__(
        self,
        in_dec_filters,
        in_enc_filters,
        out_filters,
        abn_block=ABN,
        pre_dropout_rate=0.0,
        post_dropout_rate=0.0,
        **kwargs
    ):
        super(UnetDecoderBlock, self).__init__()

        self.pre_drop = nn.Dropout(pre_dropout_rate, inplace=True)

        self.conv1 = nn.Conv2d(
            in_dec_filters + in_enc_filters,
            out_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            **kwargs
        )
        self.bn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(
            out_filters,
            out_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            **kwargs
        )
        self.bn2 = abn_block(out_filters)

        self.post_drop = nn.Dropout(post_dropout_rate, inplace=True)

    def forward(self, x, enc):
        lat_size = enc.size()[2:]
        x = F.interpolate(x, size=lat_size, mode="bilinear", align_corners=False)

        x = torch.cat([x, enc], 1)

        x = self.pre_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.post_drop(x)
        return x


# class UNetDecoder(DecoderModule):
#     def __init__(
#         self, features, start_features: int, dilation_factors=[1, 1, 1, 1], **kwargs
#     ):
#         super().__init__()
#         decoder_features = start_features
#         reversed_features = list(reversed(features))
#
#         output_filters = [decoder_features]
#         self.center = UnetCentralBlock(reversed_features[0], decoder_features)
#
#         if dilation_factors is None:
#             dilation_factors = [1] * len(reversed_features)
#
#         blocks = []
#         for block_index, encoder_features in enumerate(reversed_features):
#             blocks.append(
#                 UnetDecoderBlock(
#                     output_filters[-1],
#                     encoder_features,
#                     decoder_features,
#                     dilation=dilation_factors[block_index],
#                 )
#             )
#             output_filters.append(decoder_features)
#             # print(block_index, decoder_features, encoder_features, decoder_features)
#             decoder_features = decoder_features // 2
#
#         self.blocks = nn.ModuleList(blocks)
#         self.output_filters = output_filters
#
#     def forward(self, features):
#         reversed_features = list(reversed(features))
#         decoder_outputs = [self.center(reversed_features[0])]
#
#         for block_index, decoder_block, encoder_output in zip(
#             range(len(self.blocks)), self.blocks, reversed_features
#         ):
#             # print(block_index, decoder_outputs[-1].size(), encoder_output.size())
#             decoder_outputs.append(decoder_block(decoder_outputs[-1], encoder_output))
#
#         return decoder_outputs


class UNetDecoder(DecoderModule):
    def __init__(
        self, feature_maps: List[int], decoder_features: int, mask_channels: int
    ):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [
                decoder_features * (2 ** i) for i in range(len(feature_maps))
            ]

        blocks = []
        for block_index, in_enc_features in enumerate(feature_maps[:-1]):
            blocks.append(
                UnetDecoderBlock(
                    decoder_features[block_index + 1],
                    in_enc_features,
                    decoder_features[block_index],
                    mask_channels,
                )
            )

        self.center = UnetCentralBlock(
            feature_maps[-1], decoder_features[-1], mask_channels
        )
        self.blocks = nn.ModuleList(blocks)
        self.output_filters = decoder_features

    def forward(self, feature_maps):

        output, dsv = self.center(feature_maps[-1])
        decoder_outputs = [output]
        dsv_list = [dsv]

        for decoder_block, encoder_output in zip(
            reversed(self.blocks), reversed(feature_maps[:-1])
        ):
            output, dsv = decoder_block(output, encoder_output)
            decoder_outputs.append(output)
            dsv_list.append(dsv)

        dsv_list = list(reversed(dsv_list))
        decoder_outputs = list(reversed(decoder_outputs))

        return decoder_outputs, dsv_list
