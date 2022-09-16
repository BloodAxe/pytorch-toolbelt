from typing import Tuple

import torch
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
