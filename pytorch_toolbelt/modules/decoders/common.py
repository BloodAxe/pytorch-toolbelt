__all__ = ["DecoderModule"]

from torch import nn


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)
