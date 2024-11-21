from .common import GenericTimmEncoder


class MaxVitEncoder(GenericTimmEncoder):
    def __init__(self, model_name: str, pretrained=True, **kwargs):
        super().__init__(model_name, pretrained=pretrained, **kwargs)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        from pytorch_toolbelt.modules import make_n_channel_input

        self.encoder.stem.conv1 = make_n_channel_input(self.encoder.stem.conv1, input_channels)
        return self
