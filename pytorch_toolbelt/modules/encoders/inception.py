from .common import EncoderModule, _take, make_n_channel_input
from ..backbone.inceptionv4 import inceptionv4

__all__ = ["InceptionV4Encoder"]


class InceptionV4Encoder(EncoderModule):
    def __init__(self, pretrained=True, layers=None, **kwargs):
        backbone = inceptionv4(pretrained="imagenet" if pretrained else None)
        channels = [64, 192, 384, 1024, 1536]
        strides = [2, 4, 8, 16, 32]  # Note output strides are approximate
        if layers is None:
            layers = [1, 2, 3, 4]
        features = backbone.features
        super().__init__(channels, strides, layers)

        self.layer0 = features[0:3]
        self.layer1 = features[3:5]
        self.layer2 = features[5:10]
        self.layer3 = features[10:18]
        self.layer4 = features[18:22]

        self._output_strides = _take(strides, layers)
        self._output_filters = _take(channels, layers)

    def forward(self, x):
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)
            x = output

        # Return only features that were requested
        return _take(output_features, self._layers)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def change_input_channels(self, input_channels: int, mode="auto"):
        self.layer0[0].conv = make_n_channel_input(self.layer0[0].conv, input_channels, mode)
        return self
