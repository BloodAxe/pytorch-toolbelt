import math
from abc import abstractmethod
from typing import Tuple, Optional

import dataclasses

import cv2
import einops
import numpy as np
import torch
from pytorch_toolbelt.utils import count_parameters, describe_outputs, to_numpy, grid_stack
from torch import nn, Tensor

__all__ = [
    "normalized_spatial_coordinates",
    "fourier_position_encodings",
    "FourierPositionEncoding",
    "PositionEncoding",
    "PositionEncodingOutput",
]


def normalized_spatial_coordinates(spatial_shape: Tuple[int, ...], v_min=-1.0, v_max=1.0) -> Tensor:
    """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].
    :param spatial_shape:
    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(spatial_shape)).
    """
    coords = [torch.linspace(v_min, v_max, steps=s) for s in spatial_shape]
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))
    return grid

def fourier_position_encodings(
    p: Tensor,
    num_frequency_bands: int,
    max_frequencies: Optional[Tuple[int, ...]] = None,
    include_positions: bool = True,
) -> Tensor:
    """Fourier-encode positions p using self.num_bands frequency bands.

    :param p: positions of shape (*d, c) where c = len(d).
    :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
           2-tuple for images, ...). If `None` values are derived from shape of p.
    :param include_positions: whether to include input positions p in returned encodings tensor.
    :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
    """
    encodings = []

    if max_frequencies is None:
        max_frequencies = p.shape[:-1]

    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands, device=p.device) for max_freq in max_frequencies
    ]
    frequency_grids = []

    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

    if include_positions:
        encodings.append(p)

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

    return torch.cat(encodings, dim=-1)


@dataclasses.dataclass
class PositionEncodingOutput:
    position_encoding: Tensor
    encoded_input: Tensor


class PositionEncoding(nn.Module):
    @property
    @abstractmethod
    def num_output_channels(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_position_encoding_channels(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x: Tensor) -> PositionEncodingOutput:
        raise NotImplementedError()


class FourierPositionEncoding(PositionEncoding):
    def __init__(
        self,
        spatial_shape: Tuple[int, int],
        num_input_channels: int,
        num_frequency_bands: int,
        include_positions: bool,
        num_output_channels: Optional[int],
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.image_shape = spatial_shape + (num_input_channels,) # [H,W,C]

        num_position_encoding_channels = len(spatial_shape) * (2 * num_frequency_bands + include_positions)

        # create encodings for single example
        pos = normalized_spatial_coordinates(spatial_shape)
        enc = fourier_position_encodings(pos, num_frequency_bands, include_positions=include_positions)

        # p_rgbs = []
        # for i in range(enc.size(-1)):
        #     p = to_numpy(enc[..., i])
        #     p_rgb = cv2.applyColorMap( (255 * (p + 1) / 2).astype(np.uint8)  ,  cv2.COLORMAP_JET)
        #     p_rgb = cv2.resize(p_rgb, dsize=None, fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        #     cv2.imwrite(f"position_encoding_{i:03d}.png",p_rgb)
        #
        #     p_rgbs.append(p_rgb)
        #
        # cv2.imwrite(f"position_encoding.png", grid_stack(p_rgbs, cols=16))

        # flatten encodings along spatial dimensions
        enc = einops.rearrange(enc, "... c -> (...) c")

        self.register_buffer("position_encoding", enc)

        # Figure out output number of channels (image + position encoding)
        _num_output_channels = num_input_channels + num_position_encoding_channels
        if num_output_channels is not None:
            self.project = nn.Linear(_num_output_channels, num_output_channels)
            _num_output_channels = num_output_channels
        else:
            self.project = nn.Identity()

        self._num_output_channels = _num_output_channels
        self._num_position_encoding_channels = num_position_encoding_channels

    @property
    def num_output_channels(self) -> int:
        return self._num_output_channels

    @property
    def num_position_encoding_channels(self) -> int:
        return self._num_position_encoding_channels

    def forward(self, x: Tensor) -> PositionEncodingOutput:
        """

        Args:
            x: Input image of shape (B,C,H,W)

        Returns:

        """
        x = torch.moveaxis(x, 1, -1)  # Move BCHW to BHWC
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        x = einops.rearrange(x, "b ... c -> b (...) c")

        x_enc = einops.repeat(self.position_encoding, "... -> b ...", b=b)
        x = torch.cat([x, x_enc], dim=-1)
        x = self.project(x)
        return PositionEncodingOutput(position_encoding=x_enc, encoded_input=x)


if __name__ == "__main__":
    input = torch.randn((4, 64, 128, 96)).cuda()

    for pe in [
        FourierPositionEncoding(
            spatial_shape=(128, 96),
            num_input_channels=64,
            num_frequency_bands=64,
            include_positions=True,
            num_output_channels=None,
        ),
        FourierPositionEncoding(
            spatial_shape=(128, 96),
            num_input_channels=64,
            num_frequency_bands=64,
            include_positions=True,
            num_output_channels=512,
        ),
    ]:
        outputs = pe.cuda()(input)
        print(count_parameters(pe, human_friendly=True))
