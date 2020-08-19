from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from ..dsconv import DepthwiseSeparableConv2d

__all__ = ["CANDecoder"]


class RCM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.block(x) + x


def cfm_branch(in_channels: int, out_channels: int, kernel_size: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def ds_cfm_branch(in_channels: int, out_channels: int, kernel_size: int):
    return nn.Sequential(
        DepthwiseSeparableConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        DepthwiseSeparableConv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        ),
        nn.BatchNorm2d(out_channels),
    )


class CFM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=[3, 5, 7, 11]):
        super().__init__()
        self.gp_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_branches = nn.ModuleList(ds_cfm_branch(in_channels, out_channels, ks) for ks in kernel_sizes)

    def forward(self, x):
        gp = self.gp_branch(x)
        gp = gp.expand_as(x)

        conv_branches = [conv(x) for conv in self.conv_branches]
        return torch.cat(conv_branches + [gp], dim=1)


class AMM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, encoder, decoder):
        decoder = F.interpolate(decoder, size=encoder.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([encoder, decoder], dim=1)
        x = self.conv_bn_relu(x)
        x = F.adaptive_avg_pool2d(x, 1) * x
        return encoder + x


class CANDecoder(nn.Module):
    """
    Context Aggregation Network
    """

    def __init__(self, features: List[int], out_channels=256):
        super().__init__()

        self.encoder_rcm = nn.ModuleList(RCM(in_channels, out_channels) for in_channels in features)
        self.cfm = nn.Sequential(CFM(out_channels, out_channels), RCM(out_channels * 5, out_channels))

        self.amm_blocks = nn.ModuleList(AMM(out_channels, out_channels) for in_channels in features[:-1])
        self.rcm_blocks = nn.ModuleList(RCM(out_channels, out_channels) for in_channels in features[:-1])

        self.output_filters = [out_channels] * len(features)

    def forward(self, features):
        features = [rcm(x) for x, rcm in zip(features, self.encoder_rcm)]

        x = self.cfm(features[-1])
        outputs = [x]
        num_blocks = len(self.amm_blocks)
        for index in range(num_blocks):
            block_index = num_blocks - index - 1
            encoder_input = features[block_index]
            x = self.amm_blocks[block_index](encoder_input, x)
            x = self.rcm_blocks[block_index](x)
            outputs.append(x)

        return outputs[::-1]
