from collections import OrderedDict
from typing import List

from pytorch_toolbelt.modules import ACT_RELU, get_activation_block
from pytorch_toolbelt.modules.encoders import EncoderModule, make_n_channel_input
from torch import nn
import torch.nn.functional as F

__all__ = ["StackedHGEncoder"]


class HGStemBlock(nn.Module):
    def __init__(self, input_channels, output_channels, activation):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = activation(inplace=True)

        self.conv3 = nn.Conv2d(16, output_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.act3 = activation(inplace=True)

        self.conv4 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(output_channels)
        self.act4 = activation(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        return x


class HGResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, reduction=2):
        super(HGResidualBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        mid_channels = input_channels // reduction

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=False)

        if input_channels == output_channels:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip_layer(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class HGBlock(nn.Module):
    def __init__(self, depth: int, input_features: int, features, increase=0):
        super(HGBlock, self).__init__()
        nf = features + increase
        self.up1 = HGResidualBlock(input_features, features)
        # Lower branch
        self.pool1 = nn.AvgPool2d(2, 2)
        self.low1 = HGResidualBlock(input_features, nf)
        self.n = depth
        # Recursive hourglass
        if self.n > 1:
            self.low2 = HGBlock(depth - 1, nf, nf, increase=increase)
        else:
            self.low2 = HGResidualBlock(nf, nf)
        self.low3 = HGResidualBlock(nf, features)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.final = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    ("bn", nn.BatchNorm2d(features)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.upsample(low3)
        x = up1 + up2
        x = self.final(x)
        return x


class StackedHGEncoder(EncoderModule):
    """
    Original implementation: https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
    """
    def __init__(
        self, input_channels: int = 3, stack_level: int = 8, depth: int = 4, features: int = 256, activation=ACT_RELU
    ):
        super().__init__([64] + [features] * stack_level, [4] + [4] * stack_level, list(range(0, stack_level + 1)))
        self.stem = HGStemBlock(input_channels, 64, activation=get_activation_block(activation))
        input_features = 64
        modules = []
        for _ in range(stack_level):
            modules.append(HGBlock(depth, input_features, features, increase=0))
            input_features = features
        self.blocks = nn.ModuleList(modules)

    def forward(self, x):
        outputs = [self.stem(x)]
        for hourglass in self.blocks:
            outputs.append(hourglass(outputs[-1]))
        return outputs

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode)
        return self

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem] + self.blocks


class HGSupervisionBlock(nn.Module):
    def __init__(self, features, supervision_channels: int):
        super().__init__()
        self.squeeze = nn.Conv2d(features, supervision_channels, kernel_size=1)
        self.expand = nn.Conv2d(supervision_channels, features, kernel_size=1)

    def forward(self, x):
        sup_mask = self.squeeze(x)
        sup_features = self.expand(sup_mask)
        return sup_mask, sup_features


class StackedSupervisedHGEncoder(EncoderModule):
    def __init__(
        self,
        input_channels: int,
        supervision_channels: int,
        stack_level: int = 8,
        depth: int = 4,
        features: int = 256,
        activation=ACT_RELU,
    ):
        super().__init__(
            [features] + [features] * stack_level, [4] + [4] * stack_level, list(range(0, stack_level + 1))
        )
        self.stem = HGStemBlock(input_channels, features, activation=get_activation_block(activation))
        input_features = features
        modules = []
        for _ in range(stack_level):
            modules.append(HGBlock(depth, input_features, features, increase=0))
            input_features = features
        self.blocks = nn.ModuleList(modules)
        self.num_blocks = len(modules)
        self.supervision_blocks = nn.ModuleList(
            [HGSupervisionBlock(features, supervision_channels) for _ in range(self.num_blocks - 1)]
        )

    def forward(self, x):
        outputs = [self.stem(x)]
        supervision = []

        for i, hourglass in enumerate(self.blocks):
            x = outputs[-1]
            hg = hourglass(x)
            y = hg

            # Do not use supervision of last HG
            if i < self.num_blocks - 1:
                sup_mask, sup_features = self.supervision_blocks[i](x)
                supervision.append(sup_mask)
                y = y + sup_features

            outputs.append(y)

        return outputs, supervision

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode)
        return self

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem] + self.blocks
