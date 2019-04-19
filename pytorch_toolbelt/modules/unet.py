import torch
import torch.nn as nn
import torch.nn.functional as F

from .abn import ABN, ACT_RELU


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, abn_block=ABN, activation=ACT_RELU, stride=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False, **kwargs)
        self.bn1 = abn_block(out_filters, activation=activation)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, stride=stride, bias=False, **kwargs)
        self.bn2 = abn_block(out_filters, activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UnetCentralBlock(nn.Module):
    def __init__(self, in_dec_filters, out_filters, abn_block=ABN, activation=ACT_RELU, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dec_filters, out_filters, kernel_size=3, padding=1, stride=2, bias=False, **kwargs)
        self.bn1 = abn_block(out_filters, activation=activation)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1, bias=False, **kwargs)
        self.bn2 = abn_block(out_filters, activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UnetDecoderBlock(nn.Module):
    """
    """

    def __init__(self, in_dec_filters, in_enc_filters, out_filters, abn_block=ABN, activation=ACT_RELU, pre_dropout_rate=0., post_dropout_rate=0., **kwargs):
        super(UnetDecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_dec_filters + in_enc_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn1 = abn_block(out_filters, activation=activation)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn2 = abn_block(out_filters, activation=activation)

        self.pre_drop = nn.Dropout(pre_dropout_rate, inplace=True)
        self.post_drop = nn.Dropout(post_dropout_rate, inplace=True)

    def forward(self, x, enc):
        lat_size = enc.size()[2:]
        x = F.interpolate(x, size=lat_size, mode='bilinear', align_corners=True)

        x = torch.cat([x, enc], 1)

        x = self.pre_drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.post_drop(x)
        return x
