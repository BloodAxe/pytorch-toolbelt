from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..modules.activations import ABN

__all__ = ["UnetBlock", "UnetCentralBlock", "UnetDecoderBlock"]


class UnetBlock(nn.Module):
    """
    Vanilla U-Net block containing of two convolutions interleaved with batch-norm and RELU
    """

    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn1 = abn_block(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class UnetCentralBlock(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class UnetDecoderBlock(nn.Module):
    """
    """

    def __init__(
        self,
        in_dec_filters: int,
        in_enc_filters: int,
        out_filters: int,
        abn_block=ABN,
        dropout_rate=0.0,
        scale_factor=None,
        scale_mode="nearest",
        align_corners=None,
    ):
        super(UnetDecoderBlock, self).__init__()

        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self.align_corners = align_corners

        self.conv1 = nn.Conv2d(in_dec_filters + in_enc_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(out_filters)

        self.drop = nn.Dropout2d(dropout_rate, inplace=False)

        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(out_filters)

    def forward(self, x: torch.Tensor, enc: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.scale_factor is not None:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=self.align_corners
            )
        else:
            lat_size = enc.size()[2:]
            x = F.interpolate(x, size=lat_size, mode=self.scale_mode, align_corners=self.align_corners)

        if enc is not None:
            x = torch.cat([x, enc], dim=1)

        x = self.conv1(x)
        x = self.abn1(x)

        x = self.drop(x)

        x = self.conv2(x)
        x = self.abn2(x)

        return x
