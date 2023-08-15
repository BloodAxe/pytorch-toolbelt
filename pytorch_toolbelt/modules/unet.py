from torch import nn

from pytorch_toolbelt.modules.drop_path import DropPath
from pytorch_toolbelt.modules.activations import ACT_RELU, instantiate_activation_block
from pytorch_toolbelt.modules.normalization import NORM_BATCH, instantiate_normalization_block

__all__ = ["UnetBlock", "UnetResidualBlock"]


class UnetBlock(nn.Module):
    """
    Vanilla U-Net block containing of two convolutions interleaved with batch-norm and RELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation=ACT_RELU,
        normalization=NORM_BATCH,
        normalization_kwargs=None,
        activation_kwargs=None,
    ):
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {"inplace": True}
        if normalization_kwargs is None:
            normalization_kwargs = {}

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm1 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act1 = instantiate_activation_block(activation, **activation_kwargs)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm2 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act2 = instantiate_activation_block(activation, **activation_kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class UnetResidualBlock(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation=ACT_RELU,
        normalization=NORM_BATCH,
        normalization_kwargs=None,
        activation_kwargs=None,
        drop_path_rate=0.0,
    ):
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {"inplace": True}
        if normalization_kwargs is None:
            normalization_kwargs = {}

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm1 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act1 = instantiate_activation_block(activation, **activation_kwargs)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm2 = instantiate_normalization_block(normalization, out_channels, **normalization_kwargs)
        self.act2 = instantiate_activation_block(activation, **activation_kwargs)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return self.drop_path(x) + residual
