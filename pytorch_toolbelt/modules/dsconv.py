from torch import nn, Tensor

from pytorch_toolbelt.modules.activations import instantiate_activation_block

__all__ = ["DepthwiseSeparableConv2d", "DepthwiseSeparableConv2dBlock"]


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depth-wise separable convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
            bias=bias,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthwiseSeparableConv2dBlock(nn.Module):
    """
    Depthwise seperable convolution with batchnorm and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        kernel_size: int = 3,
        stride=1,
        padding=1,
        dilation=1,
    ):
        super(DepthwiseSeparableConv2dBlock, self).__init__()
        self.depthwise = DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = instantiate_activation_block(activation, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.bn(x)
        return self.act(x)
