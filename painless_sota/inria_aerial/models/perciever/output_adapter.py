from typing import Tuple, Optional

import einops
import torch
from painless_sota.inria_aerial.models.perciever.input_adapter import (
    normalized_spatial_coordinates,
    fourier_position_encodings,
)
from torch import nn, Tensor


class OutputAdapter(nn.Module):
    def __init__(self):
        """Transforms generic decoder cross-attention output to task-specific output."""
        super().__init__()

    @property
    def num_output_query_channels(self):
        raise NotImplementedError()

    def output_query(self, x, z):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class SameInputQuerySegmentationOutputAdapter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_shape: Tuple[int, int, int],
        num_output_query_channels: int,
    ):
        super().__init__()
        self._num_output_query_channels = num_output_query_channels
        self.depth2space = nn.PixelShuffle(4)
        self.linear = nn.Linear(num_output_query_channels, num_classes * 4 * 4)

        image_shape_down = (image_shape[0] // 4, image_shape[1] // 4, image_shape[2] * 4 * 4)
        *self.spatial_shape, num_image_channels = image_shape_down

    @property
    def num_output_query_channels(self):
        return self._num_output_query_channels

    def output_query(self, x: Tensor, z: Tensor) -> Tensor:
        return x

    def forward(self, x):
        y = self.linear(x)
        b, spatial_flatten, channels = y.shape

        output = torch.moveaxis(y.view([b] + self.spatial_shape + [channels]), -1, 1)
        output = self.depth2space(output)
        return output


class FourierPEQuerySegmentationOutputAdapter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_frequency_bands: int,
        image_shape: Tuple[int, int, int],
        num_output_query_channels: Optional[int],
        include_positions: bool,
    ):
        super().__init__()

        image_shape_down = (image_shape[0] // 4, image_shape[1] // 4, image_shape[2] * 4 * 4)
        *self.spatial_shape, num_image_channels = image_shape_down

        num_position_encoding_channels = len(self.spatial_shape) * (2 * num_frequency_bands + include_positions)

        # create encodings for single example
        pos = normalized_spatial_coordinates(self.spatial_shape)
        enc = fourier_position_encodings(
            pos, num_frequency_bands=num_frequency_bands, include_positions=include_positions
        )
        # flatten encodings along spatial dimensions
        enc = einops.rearrange(enc, "... c -> (...) c")

        self.register_buffer("position_encoding", enc)

        if num_output_query_channels is not None:
            self.position_encoding_project = nn.Linear(num_position_encoding_channels, num_output_query_channels)
            self._num_output_query_channels = num_output_query_channels
        else:
            self.position_encoding_project = nn.Identity()
            self._num_output_query_channels = num_position_encoding_channels

        self.linear = nn.Linear(self.num_output_query_channels, num_classes * 4 * 4)
        self.depth2space = nn.PixelShuffle(4)

    @property
    def num_output_query_channels(self):
        return self._num_output_query_channels

    def output_query(self, x: Tensor, z: Tensor) -> Tensor:
        b, *d = x.shape
        position_encoding = einops.repeat(self.position_encoding, "... -> b ...", b=b)
        return self.position_encoding_project(position_encoding)

    def forward(self, x):
        y = self.linear(x)
        b, spatial_flatten, channels = y.shape

        output = torch.moveaxis(y.view([b] + self.spatial_shape + [channels]), -1, 1)
        output = self.depth2space(output)
        return output
