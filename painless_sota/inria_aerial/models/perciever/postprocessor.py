from typing import Tuple, Optional

import torch
from pytorch_toolbelt.modules import instantiate_activation_block
from torch import nn, Tensor

__all__ = ["Depth2SpacePostprocessor"]


class Depth2SpacePostprocessor(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        spatial_shape: Tuple[int, int],
        num_output_channels: int,
        factor: int,
        output_name: Optional[str],
        activation: str,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_input_channels),
            instantiate_activation_block(activation),
            nn.Conv2d(num_input_channels, num_output_channels * factor * factor, kernel_size=3, padding=1),
        )

        self.spatial_shape = spatial_shape
        self.depth2space = nn.PixelShuffle(factor)
        self.output_name = output_name
        self.num_output_channels = num_output_channels

    def forward(self, x: Tensor):
        """

        Args:
            x: Tensor Input sequence (B, Seq, Channels)

        Returns:
            Tensor of (B, self.num_output_channels
        """
        b, spatial_flatten, channels = x.shape

        y = x.view((b,) + self.spatial_shape + (channels,))
        y = torch.moveaxis(y, -1, 1)
        y = self.project(y)

        output = self.depth2space(y)

        if self.output_name is not None:
            output = {self.output_name: output}
        return output
