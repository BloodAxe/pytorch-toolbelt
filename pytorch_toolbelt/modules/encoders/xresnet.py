from collections import OrderedDict
from typing import List, Union

import torch
from torch import nn

from pytorch_toolbelt.modules import ACT_RELU, instantiate_activation_block, ChannelSpatialGate2d
from pytorch_toolbelt.modules.encoders import EncoderModule, make_n_channel_input

__all__ = [
    "XResNet18Encoder",
    "XResNet34Encoder",
    "XResNet50Encoder",
    "XResNet101Encoder",
    "XResNet152Encoder",
    "SEXResNet18Encoder",
    "SEXResNet34Encoder",
    "SEXResNet50Encoder",
    "SEXResNet101Encoder",
    "SEXResNet152Encoder",
]


def make_conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    zero_batch_norm: bool = False,
    use_activation: bool = True,
    activation: str = ACT_RELU,
) -> torch.nn.Sequential:
    """
    Create a nn.Conv2d block followed by nn.BatchNorm2d and (optional) activation block.
    """
    batch_norm = nn.BatchNorm2d(out_channels)
    # initializer batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0.0 if zero_batch_norm else 1.0)
    layers = [
        (
            "conv",
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        ),
        ("bn", batch_norm),
    ]
    if use_activation:
        activation_block = instantiate_activation_block(activation, inplace=True)
        layers.append((activation, activation_block))

    return nn.Sequential(OrderedDict(layers))


class StemBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, activation: str = ACT_RELU):
        super().__init__()
        self.conv_bn_relu_1 = make_conv_bn_act(input_channels, 8, stride=2, activation=activation)
        self.conv_bn_relu_2 = make_conv_bn_act(8, 64, activation=activation)
        self.conv_bn_relu_3 = make_conv_bn_act(64, output_channels, activation=activation)

    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        x = self.conv_bn_relu_3(x)
        return x


class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""

    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int = 1, activation: str = ACT_RELU):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [
                make_conv_bn_act(n_inputs, n_hidden, 3, stride=stride, activation=activation),
                make_conv_bn_act(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False),
            ]
        else:
            layers = [
                make_conv_bn_act(n_inputs, n_hidden, 1, activation=activation),
                make_conv_bn_act(n_hidden, n_hidden, 3, stride=stride, activation=activation),
                make_conv_bn_act(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False),
            ]

        self.convs = nn.Sequential(*layers)
        self.activation = instantiate_activation_block(activation, inplace=True)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = make_conv_bn_act(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class SEXResNetBlock(nn.Module):
    """Creates the Squeeze&Excitation + XResNet block."""

    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int = 1, activation: str = ACT_RELU):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [
                make_conv_bn_act(n_inputs, n_hidden, 3, stride=stride, activation=activation),
                make_conv_bn_act(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False),
            ]
        else:
            layers = [
                make_conv_bn_act(n_inputs, n_hidden, 1, activation=activation),
                make_conv_bn_act(n_hidden, n_hidden, 3, stride=stride, activation=activation),
                make_conv_bn_act(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False),
            ]

        self.convs = nn.Sequential(*layers)
        self.activation = instantiate_activation_block(activation, inplace=True)
        self.se = ChannelSpatialGate2d(n_filters, reduction=4)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = make_conv_bn_act(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))


class XResNet(EncoderModule):
    def __init__(
        self,
        expansion: int,
        blocks: List[int],
        input_channels: int = 3,
        activation=ACT_RELU,
        layers=None,
        first_pool: Union[nn.MaxPool2d, nn.AvgPool2d] = nn.MaxPool2d,
        pretrained=None,
        block: Union[XResNetBlock, SEXResNetBlock] = XResNetBlock,
    ):
        assert len(blocks) == 4
        if layers is None:
            layers = [1, 2, 3, 4]

        n_filters = [64 // expansion, 64, 128, 256, 512]
        channels = [64, 64 * expansion, 128 * expansion, 256 * expansion, 512 * expansion]

        super().__init__(channels, [2, 4, 8, 16, 32], layers)

        res_layers = [
            self._make_layer(
                block,
                expansion,
                n_filters[i],
                n_filters[i + 1],
                n_blocks=l,
                stride=1 if i == 0 else 2,
                activation=activation,
            )
            for i, l in enumerate(blocks)
        ]

        self.stem = StemBlock(input_channels, 64, activation=activation)
        self.layer1 = nn.Sequential(
            OrderedDict([("pool", first_pool(kernel_size=3, stride=2, padding=1)), ("block", res_layers[0])])
        )

        self.layer2 = res_layers[1]
        self.layer3 = res_layers[2]
        self.layer4 = res_layers[3]

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    @staticmethod
    def _make_layer(block, expansion, n_inputs: int, n_filters: int, n_blocks: int, stride: int, activation: str):
        return nn.Sequential(
            *[
                block(
                    expansion,
                    n_inputs if i == 0 else n_filters,
                    n_filters,
                    stride if i == 0 else 1,
                    activation=activation,
                )
                for i in range(n_blocks)
            ]
        )

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv_bn_relu_1.conv = make_n_channel_input(self.stem.conv_bn_relu_1.conv, input_channels, mode)


def XResNet18Encoder(**kwargs):
    return XResNet(1, [2, 2, 2, 2], **kwargs)


def XResNet34Encoder(**kwargs):
    return XResNet(1, [3, 4, 6, 3], **kwargs)


def XResNet50Encoder(**kwargs):
    return XResNet(4, [3, 4, 6, 3], **kwargs)


def XResNet101Encoder(**kwargs):
    return XResNet(4, [3, 4, 23, 3], **kwargs)


def XResNet152Encoder(**kwargs):
    return XResNet(4, [3, 8, 36, 3], **kwargs)


# SE-XResNet


def SEXResNet18Encoder(**kwargs):
    return XResNet(1, [2, 2, 2, 2], block=SEXResNetBlock, **kwargs)


def SEXResNet34Encoder(**kwargs):
    return XResNet(1, [3, 4, 6, 3], block=SEXResNetBlock, **kwargs)


def SEXResNet50Encoder(**kwargs):
    return XResNet(4, [3, 4, 6, 3], block=SEXResNetBlock, **kwargs)


def SEXResNet101Encoder(**kwargs):
    return XResNet(4, [3, 4, 23, 3], block=SEXResNetBlock, **kwargs)


def SEXResNet152Encoder(**kwargs):
    return XResNet(4, [3, 8, 36, 3], block=SEXResNetBlock, **kwargs)
