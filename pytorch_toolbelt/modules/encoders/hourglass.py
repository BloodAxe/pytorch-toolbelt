from collections import OrderedDict
from typing import List

from pytorch_toolbelt.modules import ACT_RELU, get_activation_block
from pytorch_toolbelt.modules.encoders import EncoderModule, make_n_channel_input
from torch import nn
import torch.nn.functional as F

__all__ = ["StackedHGEncoder", "StackedSupervisedHGEncoder"]


def conv1x1_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU(inplace=True)),
            ]
        )
    )


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
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=True)

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


class HGStemBlock(nn.Module):
    def __init__(self, input_channels, output_channels, activation):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = activation(inplace=True)

        self.residual1 = HGResidualBlock(16, 32)

        self.conv3 = nn.Conv2d(32, output_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.act3 = activation(inplace=True)

        self.residual2 = HGResidualBlock(output_channels, output_channels)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.residual1(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.residual2(x)
        return x


class HGBlock(nn.Module):
    def __init__(self, depth: int, input_features: int, features, increase=0):
        super(HGBlock, self).__init__()
        nf = features + increase
        self.up1 = HGResidualBlock(input_features, features)
        # Lower branch
        self.down = nn.AvgPool2d(2, 2)
        self.low1 = HGResidualBlock(input_features, nf)
        self.n = depth
        # Recursive hourglass
        if self.n > 1:
            self.low2 = HGBlock(depth - 1, nf, nf, increase=increase)
        else:
            self.low2 = HGResidualBlock(nf, nf)
        self.low3 = HGResidualBlock(nf, features)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.down(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        hg = up1 + up2
        return hg


class HGFeaturesBlock(nn.Module):
    def __init__(self, features: int, blocks=4):
        super().__init__()
        residual_blocks = [HGResidualBlock(features, features) for _ in range(blocks)]
        self.residuals = nn.Sequential(*residual_blocks)
        self.linear = conv1x1_bn_relu(features, features)

    def forward(self, x):
        x = self.residuals(x)
        x = self.linear(x)
        return x


class HGSupervisionBlock(nn.Module):
    def __init__(self, features, supervision_channels: int):
        super().__init__()
        self.squeeze = nn.Conv2d(features, supervision_channels, kernel_size=1)
        self.expand = nn.Conv2d(supervision_channels, features, kernel_size=1)

    def forward(self, x):
        sup_mask = self.squeeze(x)
        sup_features = self.expand(sup_mask)
        return sup_mask, sup_features


class StackedHGEncoder(EncoderModule):
    """
    Original implementation: https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
    """

    def __init__(
        self, input_channels: int = 3, stack_level: int = 8, depth: int = 4, features: int = 256, activation=ACT_RELU
    ):
        super().__init__(
            channels=[features] + [features] * stack_level,
            strides=[4] + [4] * stack_level,
            layers=list(range(0, stack_level + 1)),
        )
        self.stem = HGStemBlock(input_channels, features, activation=get_activation_block(activation))

        input_features = features
        modules = []

        for _ in range(stack_level):
            modules.append(HGBlock(depth, input_features, features, increase=0))
            input_features = features

        self.num_blocks = len(modules)
        self.blocks = nn.ModuleList(modules)
        self.features = nn.ModuleList([HGFeaturesBlock(features, 4) for _ in range(stack_level)])
        self.merge_features = nn.ModuleList(
            [nn.Conv2d(features, features, kernel_size=1) for _ in range(stack_level - 1)]
        )

    def forward(self, x):
        x = self.stem(x)
        outputs = [x]

        for i, hourglass in enumerate(self.blocks):
            features = self.features[i](hourglass(x))
            outputs.append(features)

            if i < self.num_blocks - 1:
                x = x + self.merge_features[i](features)

        return outputs

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode)
        return self

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem] + list(self.blocks)


class StackedSupervisedHGEncoder(StackedHGEncoder):
    def __init__(
        self,
        supervision_channels: int,
        input_channels: int = 3,
        stack_level: int = 8,
        depth: int = 4,
        features: int = 256,
        activation=ACT_RELU,
    ):
        super().__init__(input_channels, stack_level, depth, features, activation)

        self.supervision_blocks = nn.ModuleList(
            [HGSupervisionBlock(features, supervision_channels) for _ in range(stack_level - 1)]
        )

    def forward(self, x):
        x = self.stem(x)
        outputs = [x]
        supervision = []

        for i, hourglass in enumerate(self.blocks):
            features = self.features[i](hourglass(x))
            outputs.append(features)

            if i < self.num_blocks - 1:
                sup_mask, sup_features = self.supervision_blocks[i](features)
                supervision.append(sup_mask)
                x = x + self.merge_features[i](features) + sup_features

        return outputs, supervision
