from collections import OrderedDict

from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, reduction=2):
        super(ResidualBlock, self).__init__()
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


class HourGlass(nn.Module):
    def __init__(self, depth: int, input_features: int, features, bn=None, increase=0):
        super(HourGlass, self).__init__()
        nf = features + increase
        self.up1 = ResidualBlock(input_features, features)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResidualBlock(input_features, nf)
        self.n = depth
        # Recursive hourglass
        if self.n > 1:
            self.low2 = HourGlass(depth - 1, nf, nf, bn=bn, increase=increase)
        else:
            self.low2 = ResidualBlock(nf, nf)
        self.low3 = ResidualBlock(nf, features)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

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
        up2 = self.up2(low3)
        x = up1 + up2
        x = self.final(x)
        return x


class StackedHourglass(nn.Module):
    def __init__(self, stack, depth, input_features, features, bn=None, increase=0):
        super().__init__()
        modules = []
        for _ in range(stack):
            m = HourGlass(depth, input_features, features, bn, increase)
            input_features = features
            modules.append(m)

        self.blocks = nn.ModuleList(modules)

    def forward(self, x):
        for m in self.blocks:
            x = m(x)
        return x


class StackedHourGlassEncoder(EncoderModule):
    def __init__(self, input_channels:int, stack_level:int, features:int):
        self.stem = Stem