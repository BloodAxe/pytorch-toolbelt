from __future__ import absolute_import
import torch

from torch import nn
from torch.nn import functional as F

from .scse import ChannelSpatialGate2dV2
from .abn import ABN, ACT_ELU, ACT_SELU


class FPNDecoderBlock(nn.Module):
    def __init__(self, encoder_features, decoder_features):
        super().__init__()
        self.bottleneck = nn.Conv2d(encoder_features, decoder_features, kernel_size=1)

    def forward(self, enc, dec=None):
        x = self.bottleneck(enc)
        if dec is not None:
            x = x + F.interpolate(dec, size=enc.size()[2:], mode='bilinear', align_corners=True)

        return x


class FPNBlock(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.0, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, padding=1 * dilation, dilation=dilation, bias=False)
        self.abn1 = ABN(output_features, output_features, activation=ACT_SELU)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn2 = ABN(output_features, output_features, activation=ACT_SELU)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.dropout(x)
        return x


class FPNBlockSCSE(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.0, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, padding=1 * dilation, dilation=dilation, bias=False)
        self.abn1 = ABN(output_features, output_features, activation=ACT_SELU)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False)
        self.abn2 = ABN(output_features, output_features, activation=ACT_SELU)
        self.scse = ChannelSpatialGate2dV2(output_features)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.scse(x)
        x = self.dropout(x)
        return x


class FPNFuse(nn.Module):
    def __init__(self, mode='bilinear', align_corners=True):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features):
        layers = []
        dst_size = features[-1].size()[-2:]

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)
