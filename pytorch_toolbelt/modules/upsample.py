from typing import Optional, List

import torch
from torch import nn, Tensor
from math import hypot

__all__ = [
    "bilinear_upsample_initializer",
    "icnr_init",
    "DepthToSpaceUpsample2d",
    "BilinearAdditiveUpsample2d",
    "DeconvolutionUpsample2d",
    "ResidualDeconvolutionUpsample2d",
]


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
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = initializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)
    return kernel


class DepthToSpaceUpsample2d(nn.Module):
    """
    NOTE: This block is not fully ready yet. Need to figure out how to correctly initialize
    default weights to have bilinear upsample identical to OpenCV results

    https://github.com/pytorch/pytorch/pull/5429
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        n = 2 ** scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * n, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

        # with torch.no_grad():
        #     self.conv.weight.data = icnr_init(
        #         self.conv.weight, upscale_factor=scale_factor, initializer=bilinear_upsample_initializer
        #     )

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.shuffle(self.conv(x))
        return x


class BilinearAdditiveUpsample2d(nn.Module):
    """
    https://arxiv.org/abs/1707.05847
    """

    def __init__(self, in_channels: int, scale_factor: int = 2, n: int = 4):
        super().__init__()
        if in_channels % n != 0:
            raise ValueError(f"Number of input channels ({in_channels})must be divisable by n ({n})")

        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.n = n

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        x = self.upsample(x)
        n, c, h, w = x.size()
        x = x.reshape(n, c // self.n, self.n, h, w).mean(2)
        return x


class DeconvolutionUpsample2d(nn.Module):
    def __init__(self, in_channels: int, n=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // n, kernel_size=3, padding=1, stride=2)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:  # skipcq: PYL-W0221
        return self.conv(x, output_size=output_size)


class ResidualDeconvolutionUpsample2d(nn.Module):
    def __init__(self, in_channels, scale_factor=2, n=4):
        if scale_factor != 2:
            raise NotImplementedError("Scale factor other than 2 is not implemented")
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // n
        self.conv = nn.ConvTranspose2d(
            in_channels, in_channels // n, kernel_size=3, padding=1, stride=scale_factor, output_padding=1
        )
        self.residual = BilinearAdditiveUpsample2d(in_channels, scale_factor=scale_factor, n=n)

    def forward(self, x: Tensor) -> Tensor:  # skipcq: PYL-W0221
        residual_up = self.residual(x)
        return self.conv(x, output_size=residual_up.size()) + residual_up
