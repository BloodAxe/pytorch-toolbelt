from abc import abstractmethod
from typing import Optional, Tuple

import torch
from painless_sota.inria_aerial.data.functional import as_tuple_of_two
from pytorch_toolbelt.modules import instantiate_activation_block, ACT_RELU
from pytorch_toolbelt.utils import count_parameters, describe_outputs
from torch import nn, Tensor

__all__ = ["ImagePreprocessor", "Space2DepthPreprocessor", "LearnableConvPreprocessor"]


class ImagePreprocessor(nn.Module):
    @property
    @abstractmethod
    def output_stride(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_output_channels(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_input_channels(self) -> int:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x: Input 4D tensor of shape (B,C,H,W]

        Returns:
            Preprocessed tensor of shape (B,Cout, H/stride, W/stride)
        """


class Space2DepthPreprocessor(ImagePreprocessor):
    def __init__(
        self,
        spatial_shape: Tuple[int, int],
        num_input_channels: int,
        factor: int,
        num_output_channels: Optional[int],
        kernel_size: int,
        with_bn: bool,
        activation: Optional[str],
    ):
        super().__init__()
        self.space2depth = nn.PixelUnshuffle(factor)
        output_channels_after_space2depth = num_input_channels * factor * factor
        if num_output_channels is None:
            self.project = nn.Identity()
            num_output_channels = output_channels_after_space2depth
        else:
            modules = [
                nn.Conv2d(
                    output_channels_after_space2depth,
                    num_output_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=(with_bn == False),
                )
            ]
            if with_bn:
                modules.append(nn.BatchNorm2d(num_output_channels))
            if activation is not None:
                modules.append(instantiate_activation_block(activation, inplace=True))
            self.project = nn.Sequential(*modules)

        self._output_stride = factor
        self._num_input_channels = num_input_channels
        self._num_output_channels = num_output_channels
        self.spatial_shape = spatial_shape

    @property
    def num_input_channels(self) -> int:
        return self._num_input_channels

    @property
    def num_output_channels(self) -> int:
        return self._num_output_channels

    @property
    def output_stride(self) -> int:
        return self._output_stride

    def forward(self, x: Tensor) -> Tensor:
        return self.project(self.space2depth(x))

    @property
    def output_spatial_shape(self) -> Tuple[int, int]:
        return tuple([size // self.output_stride for size in as_tuple_of_two(self.spatial_shape)])



class LearnableConvPreprocessor(ImagePreprocessor):
    def __init__(
        self, spatial_shape: Tuple[int, int], num_input_channels: int, num_output_channels: int, activation: str
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels // 2, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_output_channels // 2),
            instantiate_activation_block(activation, inplace=True),
            nn.Conv2d(num_output_channels // 2, num_output_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_output_channels),
            instantiate_activation_block(activation, inplace=True),
        )
        self._input_channels = num_input_channels
        self._output_channels = num_output_channels
        self.spatial_shape = spatial_shape

    @property
    def num_input_channels(self) -> int:
        return self._input_channels

    @property
    def num_output_channels(self) -> int:
        return self._output_channels

    @property
    def output_stride(self) -> int:
        return 4

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

    @property
    def output_spatial_shape(self) -> Tuple[int, int]:
        return tuple([size // self.output_stride for size in as_tuple_of_two(self.spatial_shape)])


if __name__ == "__main__":
    input_channels = 3

    spatial_shape = (512, 384)
    input = torch.randn((4, input_channels) + spatial_shape).cuda()

    for preprocessor in [
        LearnableConvPreprocessor(spatial_shape, input_channels, num_output_channels=64, activation=ACT_RELU).cuda(),
        Space2DepthPreprocessor(
            spatial_shape,
            input_channels,
            factor=4,
            num_output_channels=64,
            activation=None,
            with_bn=False,
            kernel_size=1,
        ).cuda(),
        Space2DepthPreprocessor(
            spatial_shape,
            input_channels,
            factor=4,
            num_output_channels=None,
            activation=None,
            kernel_size=1,
            with_bn=False,
        ).cuda(),
        Space2DepthPreprocessor(
            spatial_shape,
            input_channels,
            factor=4,
            num_output_channels=64,
            kernel_size=3,
            activation=ACT_RELU,
            with_bn=False,
        ).cuda(),
        Space2DepthPreprocessor(
            spatial_shape,
            input_channels,
            factor=4,
            num_output_channels=64,
            with_bn=True,
            activation=ACT_RELU,
            kernel_size=3,
        ).cuda(),
    ]:
        output = preprocessor(input)
        print(count_parameters(preprocessor, human_friendly=True))
        print(describe_outputs(output))
        assert output.size(1) == preprocessor.num_output_channels
        assert output.size(2) * preprocessor.output_stride == input.size(2)
        assert output.size(3) * preprocessor.output_stride == input.size(3)

        assert output.size(2) == preprocessor.output_spatial_shape[0]
        assert output.size(3) == preprocessor.output_spatial_shape[1]
