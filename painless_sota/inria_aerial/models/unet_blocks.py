from functools import partial

import torch
from pytorch_toolbelt.modules import (
    ACT_RELU,
    UnetBlock,
    get_activation_block,
    ABN,
)
from pytorch_toolbelt.modules import DepthwiseSeparableConv2d
from timm.models.efficientnet_blocks import InvertedResidual
from timm.models.layers import EffectiveSEModule
from torch import nn

__all__ = [
    "ResidualUnetBlock",
    "IRBlock",
    "InvertedResidual",
    "DenseNetUnetBlock",
    "AdditionalEncoderStage",
    "get_unet_block",
]


def get_unet_block(block_name: str, activation=ACT_RELU):
    if block_name == "ResidualUnetBlock":
        return partial(ResidualUnetBlock, abn_block=partial(ABN, activation=activation))
    elif block_name == "IRBlock":
        return partial(IRBlock, act_block=get_activation_block(activation))
    elif block_name == "DenseNetUnetBlock":
        return partial(DenseNetUnetBlock, abn_block=partial(ABN, activation=activation))
    elif block_name == "UnetBlock":
        return partial(UnetBlock, abn_block=partial(ABN, activation=activation))
    elif block_name == "DoubleConvNeXtBlock":
        return partial(DoubleConvNeXtBlock, drop_path=0.1)
    else:
        raise RuntimeError(f"Unsupported unet block {block_name}")


class ResidualUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = DepthwiseSeparableConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.abn1 = abn_block(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_channels)
        self.conv3 = DepthwiseSeparableConv2d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.abn3 = abn_block(out_channels)

    def forward(self, x):
        residual = self.identity(x)

        x = self.conv1(x)
        x = self.abn1(x)

        x = self.conv2(x)
        x = self.abn2(x)

        x = self.conv3(x)
        x = self.abn3(x)

        return x + residual


class DenseNetUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, abn_block=ABN):
        super().__init__()
        self.conv1 = ResidualUnetBlock(in_channels, out_channels, abn_block=abn_block)
        self.conv2 = ResidualUnetBlock(in_channels + out_channels, out_channels, abn_block=abn_block)

    def forward(self, x):
        y = self.conv1(x)
        x = self.conv2(torch.cat([x, y], dim=1))
        return x


class AdditionalEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_layer=nn.ReLU):
        super().__init__()
        self.ir_block1 = InvertedResidual(in_channels, out_channels, act_layer=act_layer, stride=2)
        self.ir_block2 = InvertedResidual(
            out_channels, out_channels, act_layer=act_layer, dilation=2, se_layer=EffectiveSEModule
        )

    def forward(self, x):
        x = self.ir_block1(x)
        return self.ir_block2(x)


class IRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_block=nn.ReLU, stride=1):
        super().__init__()
        self.ir_block1 = InvertedResidual(in_channels, out_channels, act_layer=act_block, stride=stride)
        self.ir_block2 = InvertedResidual(out_channels, out_channels, act_layer=act_block, se_layer=EffectiveSEModule)

    def forward(self, x):
        x = self.ir_block1(x)
        x = self.ir_block2(x)
        return x


from timm.models.layers import DropPath
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) * torch.rsqrt_(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DoubleConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0.0):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            ConvNeXtBlock(out_channels, drop_path=drop_path),
            ConvNeXtBlock(out_channels, drop_path=drop_path),
        )

    def forward(self, x):
        return self.blocks(x)
