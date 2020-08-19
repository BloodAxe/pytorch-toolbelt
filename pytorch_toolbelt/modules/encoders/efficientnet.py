import math
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import List

import torch
from torch import nn

from .common import EncoderModule, make_n_channel_input, _take

__all__ = [
    "EfficientNetEncoder",
    "EfficientNetB0Encoder",
    "EfficientNetB1Encoder",
    "EfficientNetB2Encoder",
    "EfficientNetB3Encoder",
    "EfficientNetB4Encoder",
    "EfficientNetB5Encoder",
    "EfficientNetB6Encoder",
    "EfficientNetB7Encoder",
]

from .. import ABN, SpatialGate2d, ACT_SWISH


def round_filters(filters: int, width_coefficient, depth_divisor, min_depth) -> int:
    """
    Calculate and round number of filters based on depth multiplier.
    """
    filters *= width_coefficient
    min_depth = min_depth or depth_divisor
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_multiplier):
    """
    Round number of filters based on depth multiplier.
    """
    if not depth_multiplier:
        return repeats
    return int(math.ceil(depth_multiplier * repeats))


def drop_connect(inputs, p, training):
    """
    Drop connect implementation.
    """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class EfficientNetBlockArgs:
    def __init__(
        self,
        input_filters,
        output_filters,
        expand_ratio,
        repeats=1,
        kernel_size=3,
        stride=1,
        se_reduction=4,
        dropout=0.0,
        id_skip=True,
    ):
        self.in_channels = input_filters
        self.out_channels = output_filters
        self.expand_ratio = expand_ratio
        self.num_repeat = repeats
        self.se_reduction = se_reduction
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride
        self.width_coefficient = 1.0
        self.depth_coefficient = 1.0
        self.depth_divisor = 8
        self.min_filters = None
        self.id_skip = id_skip

    def __repr__(self):
        """Encode a block args class to a string representation."""
        args = [
            "r%d" % self.num_repeat,
            "k%d" % self.kernel_size,
            "s%d" % self.stride,
            "e%s" % self.expand_ratio,
            "i%d" % self.in_channels,
            "o%d" % self.out_channels,
        ]
        if self.se_reduction > 0:
            args.append("se%s" % self.se_reduction)
        return "_".join(args)

    def copy(self):
        return deepcopy(self)

    def scale(
        self, width_coefficient: float, depth_coefficient: float, depth_divisor: float = 8.0, min_filters: int = None
    ):
        copy = self.copy()
        copy.in_channels = round_filters(self.in_channels, width_coefficient, depth_divisor, min_filters)
        copy.out_channels = round_filters(self.out_channels, width_coefficient, depth_divisor, min_filters)
        copy.num_repeat = round_repeats(self.num_repeat, depth_coefficient)
        copy.width_coefficient = width_coefficient
        copy.depth_coefficient = depth_coefficient
        copy.depth_divisor = depth_divisor
        copy.min_filters = min_filters
        return copy

    @staticmethod
    def B0():
        params = get_default_efficientnet_params(dropout=0.2)
        params = [p.scale(width_coefficient=1.0, depth_coefficient=1.0) for p in params]
        return params

    @staticmethod
    def B1():
        params = get_default_efficientnet_params(dropout=0.2)
        params = [p.scale(width_coefficient=1.0, depth_coefficient=1.1) for p in params]
        return params

    @staticmethod
    def B2():
        params = get_default_efficientnet_params(dropout=0.3)
        params = [p.scale(width_coefficient=1.1, depth_coefficient=1.2) for p in params]
        return params

    @staticmethod
    def B3():
        params = get_default_efficientnet_params(dropout=0.3)
        params = [p.scale(width_coefficient=1.2, depth_coefficient=1.4) for p in params]
        return params

    @staticmethod
    def B4():
        params = get_default_efficientnet_params(dropout=0.4)
        params = [p.scale(width_coefficient=1.4, depth_coefficient=1.8) for p in params]
        return params

    @staticmethod
    def B5():
        params = get_default_efficientnet_params(dropout=0.4)
        params = [p.scale(width_coefficient=1.6, depth_coefficient=2.2) for p in params]
        return params

    @staticmethod
    def B6():
        params = get_default_efficientnet_params(dropout=0.5)
        params = [p.scale(width_coefficient=1.8, depth_coefficient=2.6) for p in params]
        return params

    @staticmethod
    def B7():
        params = get_default_efficientnet_params(dropout=0.5)
        params = [p.scale(width_coefficient=2.0, depth_coefficient=3.1) for p in params]
        return params


def get_default_efficientnet_params(dropout=0.2) -> List[EfficientNetBlockArgs]:
    #  _DEFAULT_BLOCKS_ARGS = [
    #     'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    #     'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    #     'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    #     'r1_k3_s11_e6_i192_o320_se0.25',
    # ]
    return [
        EfficientNetBlockArgs(
            repeats=1, kernel_size=3, stride=1, expand_ratio=1, input_filters=32, output_filters=16, dropout=dropout,
        ),
        EfficientNetBlockArgs(
            repeats=2, kernel_size=3, stride=2, expand_ratio=6, input_filters=16, output_filters=24, dropout=dropout,
        ),
        EfficientNetBlockArgs(
            repeats=2, kernel_size=5, stride=2, expand_ratio=6, input_filters=24, output_filters=40, dropout=dropout,
        ),
        EfficientNetBlockArgs(
            repeats=3, kernel_size=3, stride=2, expand_ratio=6, input_filters=40, output_filters=80, dropout=dropout,
        ),
        EfficientNetBlockArgs(
            repeats=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=80, output_filters=112, dropout=dropout,
        ),
        EfficientNetBlockArgs(
            repeats=4, kernel_size=5, stride=2, expand_ratio=6, input_filters=112, output_filters=192, dropout=dropout,
        ),
        EfficientNetBlockArgs(
            repeats=1, kernel_size=3, stride=1, expand_ratio=6, input_filters=192, output_filters=320, dropout=dropout,
        ),
    ]


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args: EfficientNetBlockArgs, abn_block: ABN):
        super().__init__()

        self.has_se = block_args.se_reduction is not None
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.expand_ratio = block_args.expand_ratio
        self.stride = block_args.stride

        # Expansion phase
        inp = block_args.in_channels  # number of input channels
        oup = block_args.in_channels * block_args.expand_ratio  # number of output channels

        if block_args.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.abn0 = abn_block(oup)

        # Depthwise convolution phase
        self.depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=block_args.kernel_size,
            padding=block_args.kernel_size // 2,
            stride=block_args.stride,
            bias=False,
        )
        self.abn1 = abn_block(oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            se_channels = max(1, inp // block_args.se_reduction)
            self.se_block = SpatialGate2d(oup, squeeze_channels=se_channels)

        # Output phase
        self.project_conv = nn.Conv2d(in_channels=oup, out_channels=block_args.out_channels, kernel_size=1, bias=False)
        self.abn2 = abn_block(block_args.out_channels)

        self.input_filters = block_args.in_channels
        self.output_filters = block_args.out_channels

        self.reset_parameters()

    def reset_parameters(self):
        pass

    #     if hasattr(self, "expand_conv"):
    #         torch.nn.init.kaiming_uniform_(
    #             self.expand_conv.weight,
    #             a=abn_params.get("slope", 0),
    #             nonlinearity=sanitize_activation_name(self.abn2["activation"]),
    #         )
    #
    #     torch.nn.init.kaiming_uniform_(
    #         self.depthwise_conv.weight,
    #         a=abn_params.get("slope", 0),
    #         nonlinearity=sanitize_activation_name(abn_params["activation"]),
    #     )
    #     torch.nn.init.kaiming_uniform_(
    #         self.project_conv.weight,
    #         a=abn_params.get("slope", 0),
    #         nonlinearity=sanitize_activation_name(abn_params["activation"]),
    #     )

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = inputs
        if self.expand_ratio != 1:
            # Expansion and Depthwise Convolution
            x = self.abn0(self.expand_conv(inputs))

        x = self.abn1(self.depthwise_conv(x))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se_block(x)

        x = self.abn2(self.project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNetStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block: ABN):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.abn = abn_block(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.abn(x)
        return x


class EfficientNetEncoder(EncoderModule):
    @staticmethod
    def build_layer(block_args: List[EfficientNetBlockArgs], abn_block: ABN):
        blocks = []
        for block_index, cfg in enumerate(block_args):
            module = []
            # The first block needs to take care of stride and filter size increase.
            module.append(("mbconv_0", MBConvBlock(cfg, abn_block)))

            if cfg.num_repeat > 1:
                cfg = cfg.copy()
                cfg.stride = 1
                cfg.in_channels = cfg.out_channels

                for i in range(cfg.num_repeat - 1):
                    module.append((f"mbconv_{i+1}", MBConvBlock(cfg, abn_block)))

            module = nn.Sequential(OrderedDict(module))
            blocks.append((f"block_{block_index}", module))

        return nn.Sequential(OrderedDict(blocks))

    def __init__(self, encoder_config: List[EfficientNetBlockArgs], in_channels=3, activation=ACT_SWISH, layers=None):
        if layers is None:
            layers = [1, 2, 3, 4]

        blocks2layers = [
            # Layer 0
            [encoder_config[0], encoder_config[1]],
            # Layer 1
            [encoder_config[2]],
            # Layer 2
            [encoder_config[3], encoder_config[4]],
            # Layer 3
            [encoder_config[5], encoder_config[6]],
        ]

        filters = [encoder_config[0].in_channels] + [cfg[-1].out_channels for cfg in blocks2layers]
        strides = [2, 4, 8, 16, 32]
        abn_block = partial(ABN, activation=activation)
        super().__init__(filters, strides, layers)

        self.stem = EfficientNetStem(in_channels, encoder_config[0].in_channels, abn_block)
        self.layers = nn.ModuleList([self.build_layer(cfg, abn_block) for cfg in blocks2layers])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # skipcq: PYL-W0221
        x = self.stem(x)
        output_features = [x]
        for layer in self.layers:
            output = layer(x)
            output_features.append(output)
            x = output
        # Return only features that were requested
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv = make_n_channel_input(self.stem.conv, input_channels, mode)
        return self


class EfficientNetB0Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B0(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB1Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B1(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB2Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B2(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB3Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B3(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB4Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B4(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB5Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B5(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB6Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B6(), in_channels=in_channels, activation=activation, layers=layers)


class EfficientNetB7Encoder(EfficientNetEncoder):
    def __init__(self, in_channels=3, activation=ACT_SWISH, layers=None):
        super().__init__(EfficientNetBlockArgs.B7(), in_channels=in_channels, activation=activation, layers=layers)
