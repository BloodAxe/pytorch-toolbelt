import inspect
from collections import OrderedDict
from typing import List, Callable, Tuple

import torch

from pytorch_toolbelt.modules import ACT_RELU, get_activation_block
from pytorch_toolbelt.modules.encoders import EncoderModule, make_n_channel_input
from torch import nn, Tensor

__all__ = ["StackedHGEncoder", "StackedSupervisedHGEncoder"]


def conv1x1_bn_act(in_channels, out_channels, activation=nn.ReLU):
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("act", activation(inplace=True)),
            ]
        )
    )


class HGResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, reduction=2, activation: Callable = nn.ReLU):
        super(HGResidualBlock, self).__init__()

        mid_channels = input_channels // reduction

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.act1 = activation(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = activation(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.act3 = activation(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=True)

        if input_channels == output_channels:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1)
            torch.nn.init.zeros_(self.skip_layer.bias)

        torch.nn.init.zeros_(self.conv3.bias)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        residual = self.skip_layer(x)

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)
        out += residual
        return out


class HGStemBlock(nn.Module):
    def __init__(self, input_channels, output_channels, activation: Callable = nn.ReLU):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = activation(inplace=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = activation(inplace=True)

        self.residual1 = HGResidualBlock(64, 128)
        self.residual2 = HGResidualBlock(128, output_channels)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class HGBlock(nn.Module):
    """
    A single Hourglass model block.
    """

    def __init__(
        self,
        depth: int,
        input_features: int,
        features,
        increase=0,
        activation=nn.ReLU,
        repeats=1,
        pooling_block=nn.MaxPool2d,
    ):
        super(HGBlock, self).__init__()
        nf = features + increase

        if inspect.isclass(pooling_block) and issubclass(pooling_block, (nn.MaxPool2d, nn.AvgPool2d)):
            self.down = pooling_block(kernel_size=2, padding=0, stride=2)
        else:
            self.down = pooling_block(input_features)

        if repeats == 1:
            self.up1 = HGResidualBlock(input_features, features, activation=activation)
            self.low1 = HGResidualBlock(input_features, nf, activation=activation)
        else:
            up_blocks = []
            up_input_features = input_features
            for _ in range(repeats):
                up_blocks.append(HGResidualBlock(up_input_features, features))
                up_input_features = features
            self.up1 = nn.Sequential(*up_blocks)

            down_blocks = []
            down_input_features = input_features
            for _ in range(repeats):
                up_blocks.append(HGResidualBlock(down_input_features, nf))
                down_input_features = nf
            self.low1 = nn.Sequential(*down_blocks)

        self.depth = depth
        # Recursive hourglass
        if self.depth > 1:
            self.low2 = HGBlock(
                depth - 1,
                nf,
                nf,
                increase=increase,
                pooling_block=pooling_block,
                activation=activation,
                repeats=repeats,
            )
        else:
            self.low2 = HGResidualBlock(nf, nf, activation=activation)
        self.low3 = HGResidualBlock(nf, features, activation=activation)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        up1 = self.up1(x)
        pool1 = self.down(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up(low3)
        hg = up1 + up2
        return hg


class HGFeaturesBlock(nn.Module):
    def __init__(self, features: int, activation: Callable, blocks=1):
        super().__init__()
        residual_blocks = [HGResidualBlock(features, features, activation=activation) for _ in range(blocks)]
        self.residuals = nn.Sequential(*residual_blocks)
        self.linear = conv1x1_bn_act(features, features, activation=activation)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.residuals(x)
        x = self.linear(x)
        return x


class HGSupervisionBlock(nn.Module):
    def __init__(self, features, supervision_channels: int):
        super().__init__()
        self.squeeze = nn.Conv2d(features, supervision_channels, kernel_size=1)
        self.expand = nn.Conv2d(supervision_channels, features, kernel_size=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # skipcq: PYL-W0221
        sup_mask = self.squeeze(x)
        sup_features = self.expand(sup_mask)
        return sup_mask, sup_features


class StackedHGEncoder(EncoderModule):
    """
    Original implementation: https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
    """

    def __init__(
        self,
        input_channels: int = 3,
        stack_level: int = 8,
        depth: int = 4,
        features: int = 256,
        activation=ACT_RELU,
        repeats=1,
        pooling_block=nn.MaxPool2d,
    ):
        super().__init__(
            channels=[features] + [features] * stack_level,
            strides=[4] + [4] * stack_level,
            layers=list(range(0, stack_level + 1)),
        )

        self.stack_level = stack_level
        self.depth_level = depth
        self.num_features = features

        act = get_activation_block(activation)
        self.stem = HGStemBlock(input_channels, features, activation=act)

        input_features = features
        modules = []

        for _ in range(stack_level):
            modules.append(
                HGBlock(
                    depth,
                    input_features,
                    features,
                    increase=0,
                    activation=act,
                    repeats=repeats,
                    pooling_block=pooling_block,
                )
            )
            input_features = features

        self.num_blocks = len(modules)
        self.blocks = nn.ModuleList(modules)
        self.features = nn.ModuleList(
            [HGFeaturesBlock(features, blocks=4, activation=act) for _ in range(stack_level)]
        )
        self.merge_features = nn.ModuleList(
            [nn.Conv2d(features, features, kernel_size=1) for _ in range(stack_level - 1)]
        )

    def __str__(self):
        return f"hg_s{self.stack_level}_d{self.depth_level}_f{self.num_features}"

    def forward(self, x: Tensor) -> List[Tensor]:  # skipcq: PYL-W0221
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
        repeats=1,
        pooling_block=nn.MaxPool2d,
        supervision_block=HGSupervisionBlock,
    ):
        super().__init__(
            input_channels=input_channels,
            stack_level=stack_level,
            depth=depth,
            features=features,
            activation=activation,
            repeats=repeats,
            pooling_block=pooling_block,
        )

        self.supervision_blocks = nn.ModuleList(
            [supervision_block(features, supervision_channels) for _ in range(stack_level - 1)]
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor]]:  # skipcq: PYL-W0221
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
