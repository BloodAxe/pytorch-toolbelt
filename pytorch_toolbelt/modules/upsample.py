from enum import Enum
from math import hypot
from typing import Optional, List, Tuple, Union, Type

import torch
from torch import nn, Tensor

__all__ = [
    "bilinear_upsample_initializer",
    "icnr_init",
    "AbstractResizeLayer",
    "PixelShuffle",
    "PixelShuffleWithLinear",
    "BilinearAdditiveUpsample2d",
    "DeconvolutionUpsample2d",
    "ResidualDeconvolutionUpsample2d",
    "instantiate_upsample_block",
    "UpsampleLayerType",
]


class UpsampleLayerType(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    PIXEL_SHUFFLE = "pixel_shuffle"
    PIXEL_SHUFFLE_LINEAR = "pixel_shuffle_linear"
    DECONVOLUTION = "deconv"
    RESIDUAL_DECONV = "residual_deconv"


class AbstractResizeLayer(nn.Module):
    """
    Basic class for all upsampling blocks. It forces the upsample block to have a specific
    signature of forward method.
    """

    def forward(self, x: Tensor, output_size: Union[Tuple[int, int], torch.Size]) -> Tensor:
        """

        :param x: Input feature map to resize
        :param output_size: Target output size. This serves as a hint for the upsample block.
        :return:
        """
        raise NotImplementedError


def bilinear_upsample_initializer(x):
    cc = x.size(2) // 2
    cr = x.size(3) // 2

    for i in range(x.size(2)):
        for j in range(x.size(3)):
            x[..., i, j] = hypot(cc - i, cr - j)

    y = 1 - x / x.sum(dim=(2, 3), keepdim=True)
    y = y / y.sum(dim=(2, 3), keepdim=True)
    return y


def icnr_init(tensor: torch.Tensor, upscale_factor=2, initializer=nn.init.kaiming_normal):
    """Fill the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        initializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (upscale_factor**2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = initializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor**2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)
    return kernel


class NearestNeighborResizeLayer(AbstractResizeLayer):
    def __init__(self, in_channels: int, scale_factor: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.scale_factor = scale_factor

    def forward(self, x: Tensor, output_size: Union[Tuple[int, int], torch.Size]) -> Tensor:
        return nn.functional.interpolate(x, size=output_size, mode="nearest")


class BilinearInterpolationLayer(AbstractResizeLayer):
    def __init__(self, in_channels: int, scale_factor: int, align_corners=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: Tensor, output_size: Union[Tuple[int, int], torch.Size]) -> Tensor:
        return nn.functional.interpolate(x, size=output_size, mode="bilinear", align_corners=self.align_corners)


class PixelShuffle(AbstractResizeLayer):
    """
    Depth to Space feature map upsampling that produces a spatially larger feature map but with smaller number of
    channels. Of the input channels is not divisble by scale_factor^2, an additional 1x1 convolution will be
    applied.

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: int, scale_factor: int):
        super().__init__()
        n = 2**scale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        rounded_channels = self.out_channels * n
        self.conv = (
            nn.Conv2d(rounded_channels, self.out_channels * n, kernel_size=1, padding=1, bias=False)
            if in_channels != rounded_channels
            else nn.Identity()
        )
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x: Tensor, output_size: Union[Tuple[int, int], torch.Size] = None) -> Tensor:
        x = self.shuffle(self.conv(x))
        return x


class PixelShuffleWithLinear(AbstractResizeLayer):
    """
    Depth to Space feature map upsampling that preserves the input channels.
    This block performs grouped convolution to increase number of channels followed by pixel shuffle.

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: int, scale_factor: int):
        super().__init__()
        n = scale_factor * scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels * n, kernel_size=3, padding=1, bias=False)
        self.out_channels = in_channels
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

    def forward(self, x: Tensor, output_size: Union[Tuple[int, int], torch.Size] = None) -> Tensor:
        x = self.shuffle(self.conv(x))
        return x


class BilinearAdditiveUpsample2d(AbstractResizeLayer):
    """
    https://arxiv.org/abs/1707.05847
    """

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.n = 2**scale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // self.n

        if in_channels % self.n != 0:
            raise ValueError(f"Number of input channels ({in_channels}) must be divisable by n ({self.n})")

        self.in_channels = in_channels
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:  # skipcq: PYL-W0221
        x = self.upsample(x)
        n, c, h, w = x.size()
        x = x.reshape(n, self.out_channels, self.n, h, w).mean(2)
        return x


class DeconvolutionUpsample2d(AbstractResizeLayer):
    def __init__(self, in_channels: int, scale_factor: int = 2):
        if scale_factor != 2:
            raise NotImplementedError("Scale factor other than 2 is not implemented")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:  # skipcq: PYL-W0221
        return self.conv(x, output_size=output_size)


class ResidualDeconvolutionUpsample2d(AbstractResizeLayer):
    def __init__(self, in_channels: int, scale_factor=2):
        if scale_factor != 2:
            raise NotImplementedError(
                f"Scale factor other than 2 is not implemented. Got scale factor of {scale_factor}"
            )
        super().__init__()
        n = scale_factor * scale_factor
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(
            in_channels, in_channels // n, kernel_size=3, padding=1, stride=scale_factor, output_padding=1
        )
        self.residual = BilinearAdditiveUpsample2d(in_channels, scale_factor=scale_factor)

    def forward(self, x: Tensor, output_size: Optional[List[int]]) -> Tensor:  # skipcq: PYL-W0221
        residual_up = self.residual(x)
        return self.conv(x, output_size=residual_up.size()) + residual_up


def instantiate_upsample_block(
    block: Union[str, Type[AbstractResizeLayer]], in_channels, scale_factor: int
) -> AbstractResizeLayer:
    if isinstance(block, str):
        block = UpsampleLayerType(block)

    if isinstance(block, UpsampleLayerType):
        block = {
            UpsampleLayerType.NEAREST: NearestNeighborResizeLayer,
            UpsampleLayerType.BILINEAR: BilinearInterpolationLayer,
            UpsampleLayerType.PIXEL_SHUFFLE: PixelShuffle,
            UpsampleLayerType.PIXEL_SHUFFLE_LINEAR: PixelShuffleWithLinear,
            UpsampleLayerType.DECONVOLUTION: DeconvolutionUpsample2d,
            UpsampleLayerType.RESIDUAL_DECONV: ResidualDeconvolutionUpsample2d,
        }[block]

    return block(in_channels, scale_factor=scale_factor)
