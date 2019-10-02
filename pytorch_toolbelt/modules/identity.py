from torch import nn

__all__ = ["Identity"]


class Identity(nn.Module):
    """The most useful module. A pass-through module which does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
