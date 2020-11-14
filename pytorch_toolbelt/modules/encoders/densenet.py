from collections import OrderedDict
from typing import List

from torch import nn
from torchvision.models import densenet121, densenet161, densenet169, densenet201, DenseNet

from .common import EncoderModule, _take, make_n_channel_input

__all__ = ["DenseNetEncoder", "DenseNet121Encoder", "DenseNet169Encoder", "DenseNet161Encoder", "DenseNet201Encoder"]


class DenseNetEncoder(EncoderModule):
    def __init__(
        self, densenet: DenseNet, strides: List[int], channels: List[int], layers: List[int], first_avg_pool=False
    ):
        if layers is None:
            layers = [1, 2, 3, 4]

        super().__init__(channels, strides, layers)

        def except_pool(block: nn.Module):
            del block.pool
            return block

        self.layer0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", densenet.features.conv0),
                    ("bn0", densenet.features.norm0),
                    ("act0", densenet.features.relu0),
                ]
            )
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool0 = self.avg_pool if first_avg_pool else densenet.features.pool0

        self.layer1 = nn.Sequential(densenet.features.denseblock1, except_pool(densenet.features.transition1))

        self.layer2 = nn.Sequential(densenet.features.denseblock2, except_pool(densenet.features.transition2))

        self.layer3 = nn.Sequential(densenet.features.denseblock3, except_pool(densenet.features.transition3))

        self.layer4 = nn.Sequential(densenet.features.denseblock4)

        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    @property
    def output_strides(self):
        return self._output_strides

    @property
    def output_filters(self):
        return self._output_filters

    def forward(self, x):
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)

            if layer == self.layer0:
                # Fist maxpool operator is not a part of layer0 because we want that layer0 output to have stride of 2
                output = self.pool0(output)
            else:
                output = self.avg_pool(output)

            x = output

        # Return only features that were requested
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.layer0.conv0 = make_n_channel_input(self.layer0.conv0, input_channels, mode=mode, **kwargs)
        return self


class DenseNet121Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet121(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 512, 1024]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class DenseNet161Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet161(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [96, 192, 384, 1056, 2208]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class DenseNet169Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet169(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 640, 1664]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)


class DenseNet201Encoder(DenseNetEncoder):
    def __init__(self, layers=None, pretrained=True, memory_efficient=False, first_avg_pool=False):
        densenet = densenet201(pretrained=pretrained, memory_efficient=memory_efficient)
        strides = [2, 4, 8, 16, 32]
        channels = [64, 128, 256, 896, 1920]
        super().__init__(densenet, strides, channels, layers, first_avg_pool)
