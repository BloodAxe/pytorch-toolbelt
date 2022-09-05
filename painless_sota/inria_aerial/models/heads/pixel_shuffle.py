__all__ = ["PixelShufflePredictionsHead"]

from typing import List, Optional

import torch
from torch import nn, Tensor


class PixelShufflePredictionsHead(nn.Module):
    def __init__(
        self,
        channels: List[int],
        num_classes: int,
        scale_factor: int = 4,
        output_name: Optional[str] = None,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(channels[0], num_classes * scale_factor * scale_factor, kernel_size=1),
        )
        self.up = nn.PixelShuffle(scale_factor)
        self.output_name = output_name

    def forward(self, feature_maps: List[Tensor], output_size: torch.Size):
        x = self.projection(feature_maps[0])
        output = self.up(x)

        if self.output_name is not None:
            return {self.output_name: output}
        else:
            return output
