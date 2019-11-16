import torch
from torch import nn
import torch.nn.functional as F

from .activated_batch_norm import ABN

__all__ = ["UnetEncoderBlock", "UnetCentralBlock", "UnetDecoderBlock"]


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, abn_block=ABN, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False, **kwargs)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False, **kwargs)
        self.abn2 = abn_block(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class UnetCentralBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, abn_block=ABN, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False, **kwargs)
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False, **kwargs)
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
        in_dec_filters,
        in_enc_filters,
        out_filters,
        abn_block=ABN,
        pre_dropout_rate=0.0,
        post_dropout_rate=0.0,
        scale_factor=None,
        scale_mode="nearest",
        align_corners=None,
        **kwargs,
    ):
        super(UnetDecoderBlock, self).__init__()

        self.pre_drop = nn.Dropout2d(pre_dropout_rate, inplace=True)

        self.conv1 = nn.Conv2d(
            in_dec_filters + in_enc_filters, out_filters, kernel_size=3, padding=1, bias=False, **kwargs
        )
        self.abn1 = abn_block(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False, **kwargs)
        self.abn2 = abn_block(out_filters)

        self.post_drop = nn.Dropout2d(post_dropout_rate, inplace=False)

        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        if self.scale_factor is not None:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=self.align_corners
            )
        else:
            lat_size = enc.size()[2:]
            x = F.interpolate(x, size=lat_size, mode=self.scale_mode, align_corners=self.align_corners)

        x = torch.cat([x, enc], 1)

        x = self.pre_drop(x)
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.post_drop(x)
        return x
