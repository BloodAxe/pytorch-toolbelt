# https://github.com/Randl/MobileNetV3-pytorch/blob/master/MobileNetV3.py

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from ..activations import HardSwish, HardSigmoid
from ..identity import Identity


def _make_divisible(v, divisor, min_value=None):
    """
    Ensure that all layers have a channel number that is divisible by 8

    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqEx(nn.Module):
    """Squeeze-Excitation block. Implemented in ONNX & CoreML friendly way.
    Original implementation: https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py
    """

    def __init__(self, n_features, reduction=4):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError("n_features must be divisible by reduction (default = 4)")

        self.linear1 = nn.Conv2d(n_features, n_features // reduction, kernel_size=1, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv2d(n_features // reduction, n_features, kernel_size=1, bias=True)
        self.nonlin2 = HardSigmoid(inplace=True)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, output_size=1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = x * y
        return y


class LinearBottleneck(nn.Module):
    def __init__(
        self,
        inplanes,
        outplanes,
        expplanes,
        k=3,
        stride=1,
        drop_prob=0,
        num_steps=3e5,
        start_step=0,
        activation=nn.ReLU,
        act_params={"inplace": True},
        SE=False,
    ):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes)
        self.db1 = nn.Dropout2d(drop_prob)
        # self.db1 = DropBlockScheduled(DropBlock2D(drop_prob=drop_prob, block_size=7), start_value=0.,
        #                               stop_value=drop_prob, nr_steps=num_steps, start_step=start_step)
        self.act1 = activation(**act_params)  # first does have act according to MobileNetV2

        self.conv2 = nn.Conv2d(
            expplanes, expplanes, kernel_size=k, stride=stride, padding=k // 2, bias=False, groups=expplanes
        )
        self.bn2 = nn.BatchNorm2d(expplanes)
        self.db2 = nn.Dropout2d(drop_prob)
        # self.db2 = DropBlockScheduled(DropBlock2D(drop_prob=drop_prob, block_size=7), start_value=0.,
        #                               stop_value=drop_prob, nr_steps=num_steps, start_step=start_step)
        self.act2 = activation(**act_params)

        self.se = SqEx(expplanes) if SE else Identity()

        self.conv3 = nn.Conv2d(expplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.db3 = nn.Dropout2d(drop_prob)
        # self.db3 = DropBlockScheduled(DropBlock2D(drop_prob=drop_prob, block_size=7), start_value=0.,
        #                               stop_value=drop_prob, nr_steps=num_steps, start_step=start_step)

        self.stride = stride
        self.expplanes = expplanes
        self.inplanes = inplanes
        self.outplanes = outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.db1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.db2(out)
        out = self.act2(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.db3(out)

        if self.stride == 1 and self.inplanes == self.outplanes:  # TODO: or add 1x1?
            out += residual  # No inplace if there is in-place activation before

        return out


class LastBlockLarge(nn.Module):
    def __init__(self, inplanes, num_classes, expplanes1, expplanes2):
        super(LastBlockLarge, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes1)
        self.act1 = HardSwish(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(expplanes1, expplanes2, kernel_size=1, stride=1)
        self.act2 = HardSwish(inplace=True)

        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(expplanes2, num_classes)

        self.expplanes1 = expplanes1
        self.expplanes2 = expplanes2
        self.inplanes = inplanes
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.avgpool(out)

        out = self.conv2(out)
        out = self.act2(out)

        # flatten for input to fully-connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))

        return out


class LastBlockSmall(nn.Module):
    def __init__(self, inplanes, num_classes, expplanes1, expplanes2):
        super(LastBlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, expplanes1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expplanes1)
        self.act1 = HardSwish(inplace=True)

        self.se = SqEx(expplanes1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Conv2d(expplanes1, expplanes2, kernel_size=1, stride=1, bias=False)
        self.act2 = HardSwish(inplace=True)

        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.fc = nn.Linear(expplanes2, num_classes)

        self.expplanes1 = expplanes1
        self.expplanes2 = expplanes2
        self.inplanes = inplanes
        self.num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.se(out)
        out = self.avgpool(out)

        out = self.conv2(out)
        out = self.act2(out)

        # flatten for input to fully-connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))

        return out


class MobileNetV3(nn.Module):
    """MobileNetV3 implementation.
    """

    def __init__(
        self, num_classes=1000, scale=1.0, in_channels=3, drop_prob=0.0, num_steps=3e5, start_step=0, small=False
    ):
        super(MobileNetV3, self).__init__()

        self.num_steps = num_steps
        self.start_step = start_step
        self.scale = scale
        self.num_classes = num_classes
        self.small = small

        # setting of bottlenecks blocks
        self.bottlenecks_setting_large = [
            # in, exp, out, s, k,         dp,    se,      act
            [16, 16, 16, 1, 3, 0, False, nn.ReLU],  # -> 112x112
            [16, 64, 24, 2, 3, 0, False, nn.ReLU],  # -> 56x56
            [24, 72, 24, 1, 3, 0, False, nn.ReLU],  # -> 56x56
            [24, 72, 40, 2, 5, 0, True, nn.ReLU],  # -> 28x28
            [40, 120, 40, 1, 5, 0, True, nn.ReLU],  # -> 28x28
            [40, 120, 40, 1, 5, 0, True, nn.ReLU],  # -> 28x28
            [40, 240, 80, 2, 3, drop_prob, False, HardSwish],  # -> 14x14
            [80, 200, 80, 1, 3, drop_prob, False, HardSwish],  # -> 14x14
            [80, 184, 80, 1, 3, drop_prob, False, HardSwish],  # -> 14x14
            [80, 184, 80, 1, 3, drop_prob, False, HardSwish],  # -> 14x14
            [80, 480, 112, 1, 3, drop_prob, True, HardSwish],  # -> 14x14
            [112, 672, 112, 1, 3, drop_prob, True, HardSwish],  # -> 14x14
            [112, 672, 160, 2, 5, drop_prob, True, HardSwish],  # -> 7x7
            [160, 672, 160, 1, 5, drop_prob, True, HardSwish],  # -> 7x7
            [160, 960, 160, 1, 5, drop_prob, True, HardSwish],  # -> 7x7
        ]
        self.bottlenecks_setting_small = [
            # in, exp, out, s, k,         dp,    se,      act
            [16, 64, 16, 2, 3, 0, True, nn.ReLU],  # -> 56x56
            [16, 72, 24, 2, 3, 0, False, nn.ReLU],  # -> 28x28
            [24, 88, 24, 1, 3, 0, False, nn.ReLU],  # -> 28x28
            [24, 96, 40, 2, 5, 0, True, HardSwish],  # -> 14x14
            [40, 240, 40, 1, 5, drop_prob, True, HardSwish],  # -> 14x14
            [40, 240, 40, 1, 5, drop_prob, True, HardSwish],  # -> 14x14
            [40, 120, 48, 1, 5, drop_prob, True, HardSwish],  # -> 14x14
            [48, 144, 96, 1, 5, drop_prob, True, HardSwish],  # -> 14x14
            [96, 288, 96, 2, 5, drop_prob, True, HardSwish],  # -> 7x7
            [96, 576, 96, 1, 5, drop_prob, True, HardSwish],  # -> 7x7
            [96, 576, 96, 1, 5, drop_prob, True, HardSwish],  # -> 7x7
        ]

        self.bottlenecks_setting = self.bottlenecks_setting_small if small else self.bottlenecks_setting_large
        for layer_settings in self.bottlenecks_setting:
            layer_settings[0] = _make_divisible(layer_settings[0] * self.scale, 8)
            layer_settings[1] = _make_divisible(layer_settings[1] * self.scale, 8)
            layer_settings[2] = _make_divisible(layer_settings[2] * self.scale, 8)

        self.conv1 = nn.Conv2d(
            in_channels, self.bottlenecks_setting[0][0], kernel_size=3, bias=False, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.bottlenecks_setting[0][0])
        self.act1 = HardSwish(inplace=True)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = self._make_bottlenecks()

        # Last convolution has 1280 output channels for scale <= 1
        self.last_exp2 = 1280 if self.scale <= 1 else _make_divisible(1280 * self.scale, 8)
        if small:
            self.last_exp1 = _make_divisible(576 * self.scale, 8)
            self.last_block = LastBlockSmall(
                self.bottlenecks_setting[-1][2], num_classes, self.last_exp1, self.last_exp2
            )
        else:
            self.last_exp1 = _make_divisible(960 * self.scale, 8)
            self.last_block = LastBlockLarge(
                self.bottlenecks_setting[-1][2], num_classes, self.last_exp1, self.last_exp2
            )

    def _make_bottlenecks(self):
        layers = []
        modules = OrderedDict()
        stage_name = "Bottleneck"

        # add LinearBottleneck
        for i, setup in enumerate(self.bottlenecks_setting):
            name = stage_name + "_{}".format(i)
            module = LinearBottleneck(
                setup[0],
                setup[2],
                setup[1],
                k=setup[4],
                stride=setup[3],
                drop_prob=setup[5],
                num_steps=self.num_steps,
                start_step=self.start_step,
                activation=setup[7],
                act_params={"inplace": True},
                SE=setup[6],
            )
            modules[name] = module

            if setup[3] == 2:
                layer = nn.Sequential(modules)
                layers.append(layer)
                modules = OrderedDict()

        if len(modules):
            layer = nn.Sequential(modules)
            layers.append(layer)

        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.last_block(x)
        return x
