from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

from .common import DecoderModule
from ..activations import ABN

__all__ = ["UNetDecoderV2", "UnetCentralBlockV2", "UnetDecoderBlockV2"]


class UnetCentralBlockV2(nn.Module):
    def __init__(self, in_dec_filters, out_filters, mask_channels, abn_block=ABN):
        super().__init__()
        self.bottleneck = nn.Conv2d(in_dec_filters, out_filters, kernel_size=1)

        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)
        self.dsv = nn.Conv2d(out_filters, mask_channels, kernel_size=1)

    def forward(self, x):
        x = self.bottleneck(x)

        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)

        dsv = self.dsv(x)

        return x, dsv


class UnetDecoderBlockV2(nn.Module):
    """
    """

    def __init__(
        self,
        in_dec_filters: int,
        in_enc_filters: int,
        out_filters: int,
        mask_channels: int,
        abn_block=ABN,
        pre_dropout_rate=0.0,
        post_dropout_rate=0.0,
        scale_factor=None,
        scale_mode="nearest",
        align_corners=None,
    ):
        super(UnetDecoderBlockV2, self).__init__()

        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self.align_corners = align_corners

        self.bottleneck = nn.Conv2d(in_dec_filters + in_enc_filters, out_filters, kernel_size=1)

        self.conv1 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)

        self.pre_drop = nn.Dropout2d(pre_dropout_rate, inplace=True)

        self.post_drop = nn.Dropout2d(post_dropout_rate)

        self.dsv = nn.Conv2d(out_filters, mask_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.scale_factor is not None:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=self.align_corners
            )
        else:
            lat_size = enc.size()[2:]
            x = F.interpolate(x, size=lat_size, mode=self.scale_mode, align_corners=self.align_corners)

        x = torch.cat([x, enc], 1)

        x = self.bottleneck(x)
        x = self.pre_drop(x)

        x = self.conv1(x)
        x = self.abn1(x)

        x = self.conv2(x)
        x = self.abn2(x)

        x = self.post_drop(x)

        dsv = self.dsv(x)
        return x, dsv


class UNetDecoderV2(DecoderModule):
    def __init__(self, feature_maps: List[int], decoder_features: int, mask_channels: int, dropout=0.0, abn_block=ABN):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]

        blocks = []
        for block_index, in_enc_features in enumerate(feature_maps[:-1]):
            blocks.append(
                UnetDecoderBlockV2(
                    decoder_features[block_index + 1],
                    in_enc_features,
                    decoder_features[block_index],
                    mask_channels,
                    abn_block=abn_block,
                    post_dropout_rate=dropout,
                )
            )

        self.center = UnetCentralBlockV2(feature_maps[-1], decoder_features[-1], mask_channels, abn_block=abn_block)
        self.blocks = nn.ModuleList(blocks)
        self.output_filters = decoder_features

        self.final = nn.Conv2d(decoder_features[0], mask_channels, kernel_size=1)

    def forward(self, feature_maps):

        output, dsv = self.center(feature_maps[-1])
        dsv_list = [dsv]

        for decoder_block, encoder_output in zip(reversed(self.blocks), reversed(feature_maps[:-1])):
            output, dsv = decoder_block(output, encoder_output)
            dsv_list.append(dsv)

        dsv_list = list(reversed(dsv_list))

        output = self.final(output)
        return output, dsv_list
