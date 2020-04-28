from collections import OrderedDict
from functools import partial

from torch import nn

from ..activations import ABN
from ..pooling import GlobalAvgPool2d


class IdentityResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=1, norm_act=ABN, dropout=None):
        """Identity-mapping residual block
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels, channels[0], 3, stride=stride, padding=dilation, bias=False, dilation=dilation
                    ),
                ),
                ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=dilation, bias=False, dilation=dilation),
                ),
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)),
                ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation,
                    ),
                ),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out


class WiderResNet(nn.Module):
    def __init__(self, structure, norm_act=ABN, classes=0):
        """Wider ResNet with pre-activation (identity mapping) blocks

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        """
        super(WiderResNet, self).__init__()
        self.structure = structure

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        IdentityResidualBlock(in_channels, channels[mod_id], norm_act=norm_act),
                    )
                )

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id <= 4:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict([("avg_pool", GlobalAvgPool2d()), ("fc", nn.Linear(in_channels, classes))])
            )

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(self.pool4(out))
        out = self.mod5(self.pool5(out))
        out = self.mod6(self.pool6(out))
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            out = self.classifier(out)

        return out


class WiderResNetA2(nn.Module):
    def __init__(self, structure, norm_act=ABN, classes=0, dilation=False):
        """Wider ResNet with pre-activation (identity mapping) blocks.
        This variant uses down-sampling by max-pooling in the first two blocks and by strided convolution in the others.

        Parameters
        ----------
        structure : list of int
            Number of residual blocks in each of the six modules of the network.
        norm_act : callable
            Function to create normalization / activation Module.
        classes : int
            If not `0` also include global average pooling and a fully-connected layer with `classes` outputs at the end
            of the network.
        dilation : bool
            If `True` apply dilation to the last three modules and change the down-sampling factor from 32 to 8.
        """
        super(WiderResNetA2, self).__init__()
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        # Initial layers
        self.mod1 = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False))]))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048), (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    stride = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    stride = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout2d, p=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout2d, p=0.5)
                else:
                    drop = None

                blocks.append(
                    (
                        "block%d" % (block_id + 1),
                        IdentityResidualBlock(
                            in_channels, channels[mod_id], norm_act=norm_act, stride=stride, dilation=dil, dropout=drop
                        ),
                    )
                )

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id < 2:
                self.add_module("pool%d" % (mod_id + 2), nn.MaxPool2d(3, stride=2, padding=1))
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.Sequential(
                OrderedDict([("avg_pool", GlobalAvgPool2d()), ("fc", nn.Linear(in_channels, classes))])
            )

    def forward(self, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        else:
            return out


def wider_resnet_16(num_classes=0, norm_act=ABN):
    return WiderResNet(structure=[1, 1, 1, 1, 1, 1], norm_act=norm_act, classes=num_classes)


def wider_resnet_20(num_classes=0, norm_act=ABN):
    return WiderResNet(structure=[1, 1, 1, 3, 1, 1], norm_act=norm_act, classes=num_classes)


def wider_resnet_38(num_classes=0, norm_act=ABN):
    return WiderResNet(structure=[3, 3, 6, 3, 1, 1], norm_act=norm_act, classes=num_classes)


def wider_resnet_16_a2(num_classes=0, norm_act=ABN):
    return WiderResNetA2(structure=[1, 1, 1, 1, 1, 1], norm_act=norm_act, classes=num_classes)


def wider_resnet_20_a2(num_classes=0, norm_act=ABN):
    return WiderResNetA2(structure=[1, 1, 1, 3, 1, 1], norm_act=norm_act, classes=num_classes)


def wider_resnet_38_a2(num_classes=0, norm_act=ABN):
    return WiderResNetA2(structure=[3, 3, 6, 3, 1, 1], norm_act=norm_act, classes=num_classes)
