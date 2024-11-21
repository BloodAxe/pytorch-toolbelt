from collections import OrderedDict
from typing import List

import torch
from torch import nn

from .common import GenericTimmEncoder
from ..common import EncoderModule, make_n_channel_input

__all__ = [
    "SKResNet18Encoder",
    "SKResNeXt50Encoder",
    "SWSLResNeXt101Encoder",
    "TResNetMEncoder",
    "TimmResnet152D",
    "TimmSEResnet152D",
    "TimmResnet26D",
    "TimmResnet50D",
    "TimmResnet101D",
    "TimmResnet200D",
]

from ... import ACT_RELU, get_activation_block


class TResNetMEncoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models import tresnet_m

        act_layer = get_activation_block(activation)
        encoder = tresnet_m(pretrained=pretrained, act_layer=act_layer)

        super().__init__([64, 64, 128, 1024, 2048], [4, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(encoder.body.SpaceToDepth, encoder.body.conv1)

        self.layer1 = encoder.body.layer1
        self.layer2 = encoder.body.layer2
        self.layer3 = encoder.body.layer3
        self.layer4 = encoder.body.layer4

    @property
    @torch.jit.unused
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]


class SKResNet18Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, no_first_max_pool=False, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models import skresnet18

        act_layer = get_activation_block(activation)
        encoder = skresnet18(pretrained=pretrained, features_only=True, act_layer=act_layer)
        super().__init__([64, 64, 128, 256, 512], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(
            OrderedDict([("conv1", encoder.conv1), ("bn1", encoder.bn1), ("act1", encoder.act1)])
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2) if no_first_max_pool else encoder.maxpool,
            encoder.layer1,
        )
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class SKResNeXt50Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models import skresnext50_32x4d

        act_layer = get_activation_block(activation)
        encoder = skresnext50_32x4d(pretrained=pretrained, act_layer=act_layer)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(
            OrderedDict([("conv1", encoder.conv1), ("bn1", encoder.bn1), ("act1", encoder.act1)])
        )

        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class SWSLResNeXt101Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        if layers is None:
            layers = [1, 2, 3, 4]
        from timm.models.resnet import swsl_resnext101_32x8d

        act_layer = get_activation_block(activation)
        encoder = swsl_resnext101_32x8d(pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__([64, 256, 512, 1024, 2048], [2, 4, 8, 16, 32], layers)
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", encoder.conv1),
                    ("bn1", encoder.bn1),
                    ("act1", encoder.act1),
                ]
            )
        )

        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @property
    def encoder_layers(self) -> List[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.stem.conv1 = make_n_channel_input(self.stem.conv1, input_channels, mode, **kwargs)
        return self


class TimmResnet152D(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        from timm.models.resnet import resnet152d

        act_layer = get_activation_block(activation)
        encoder = resnet152d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmSEResnet152D(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        from timm.models.resnet import seresnet152d

        act_layer = get_activation_block(activation)
        encoder = seresnet152d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet26D(GenericTimmEncoder):
    def __init__(
        self, pretrained=True, layers=None, activation=ACT_RELU, first_conv_stride_one: bool = False, **kwargs
    ):
        from timm.models.resnet import resnet26d

        act_layer = get_activation_block(activation)
        encoder = resnet26d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        if first_conv_stride_one:
            encoder.conv1[0].stride = (1, 1)
            for info in encoder.feature_info:
                info["reduction"] = info["reduction"] // 2

        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet50D(GenericTimmEncoder):
    def __init__(
        self, pretrained=True, layers=None, activation=ACT_RELU, first_conv_stride_one: bool = False, **kwargs
    ):
        from timm.models.resnet import resnet50d

        act_layer = get_activation_block(activation)
        encoder = resnet50d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        if first_conv_stride_one:
            encoder.conv1[0].stride = (1, 1)
            for info in encoder.feature_info:
                info["reduction"] = info["reduction"] // 2

        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet101D(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        from timm.models.resnet import resnet101d

        act_layer = get_activation_block(activation)
        encoder = resnet101d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self


class TimmResnet200D(GenericTimmEncoder):
    def __init__(self, pretrained=True, layers=None, activation=ACT_RELU, **kwargs):
        from timm.models.resnet import resnet200d

        act_layer = get_activation_block(activation)
        encoder = resnet200d(features_only=True, pretrained=pretrained, act_layer=act_layer, **kwargs)
        super().__init__(encoder, layers)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv1[0] = make_n_channel_input(self.encoder.conv1[0], input_channels, mode=mode, **kwargs)
        return self
