from .common import EncoderModule, _take, make_n_channel_input
from ..backbone.efficient_net import (
    efficient_net_b0,
    efficient_net_b6,
    efficient_net_b1,
    efficient_net_b2,
    efficient_net_b3,
    efficient_net_b4,
    efficient_net_b5,
    efficient_net_b7,
)

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


class EfficientNetEncoder(EncoderModule):
    def __init__(self, efficientnet, filters, strides, layers):
        if layers is None:
            layers = [1, 2, 4, 6]

        super().__init__(filters, strides, layers)

        self.stem = efficientnet.stem

        self.block0 = efficientnet.block0
        self.block1 = efficientnet.block1
        self.block2 = efficientnet.block2
        self.block3 = efficientnet.block3
        self.block4 = efficientnet.block4
        self.block5 = efficientnet.block5
        self.block6 = efficientnet.block6

    @property
    def encoder_layers(self):
        return [self.block0, self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]

    def forward(self, x):
        x = self.stem(x)

        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            x = output

        # Return only features that were requested
        return _take(output_features, self._layers)

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.stem.conv = make_n_channel_input(self.stem.conv, input_channels, mode)
        return self


class EfficientNetB0Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b0(num_classes=1, **kwargs),
            [16, 24, 40, 80, 112, 192, 320],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB1Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b1(num_classes=1, **kwargs),
            [16, 24, 40, 80, 112, 192, 320],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB2Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b2(num_classes=1, **kwargs),
            [16, 24, 48, 88, 120, 208, 352],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB3Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b3(num_classes=1, **kwargs),
            [24, 32, 48, 96, 136, 232, 384],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB4Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b4(num_classes=1, **kwargs),
            [24, 32, 56, 112, 160, 272, 448],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB5Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b5(num_classes=1, **kwargs),
            [24, 40, 64, 128, 176, 304, 512],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB6Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b6(num_classes=1, **kwargs),
            [32, 40, 72, 144, 200, 344, 576],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )


class EfficientNetB7Encoder(EfficientNetEncoder):
    def __init__(self, layers=None, **kwargs):
        super().__init__(
            efficient_net_b7(num_classes=1, **kwargs),
            [32, 48, 80, 160, 224, 384, 640],
            [2, 4, 8, 16, 16, 32, 32],
            layers,
        )
