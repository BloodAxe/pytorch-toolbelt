import math
from typing import Tuple, Optional

import einops
import torch
from pytorch_toolbelt.utils import count_parameters, describe_outputs
from torch import nn, Tensor

__all__ = ["normalized_spatial_coordinates", "fourier_position_encodings", "InputAdapter"]


def normalized_spatial_coordinates(spatial_shape: Tuple[int, ...], v_min=-1.0, v_max=1.0) -> Tensor:
    """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].
    :param spatial_shape:
    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(shape)).
    """
    coords = [torch.linspace(v_min, v_max, steps=s) for s in spatial_shape]
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))


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


class InputAdapter(nn.Module):
    def __init__(self):
        """Transforms and position-encodes task-specific input to generic encoder input."""
        super().__init__()

    @property
    def num_output_channels(self):
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class FourierPEImageInputAdapter(InputAdapter):
    def __init__(
        self,
        image_shape: Tuple[int, ...],
        num_frequency_bands: int,
        include_positions: bool,
        image_channels_before_concat: Optional[int] = None,
        num_output_channels: Optional[int] = None,
    ):
        image_shape_down = (image_shape[0] // 4, image_shape[1] // 4, image_shape[2] * 4 * 4)
        *spatial_shape, num_image_channels = image_shape_down

        num_position_encoding_channels = len(spatial_shape) * (2 * num_frequency_bands + include_positions)

        # create encodings for single example
        pos = normalized_spatial_coordinates(spatial_shape)
        enc = fourier_position_encodings(
            pos, num_frequency_bands=num_frequency_bands, include_positions=include_positions
        )
        # flatten encodings along spatial dimensions
        enc = einops.rearrange(enc, "... c -> (...) c")

        super().__init__()
        self.image_shape_down = image_shape_down
        self.num_frequency_bands = num_frequency_bands
        self.space2depth = nn.PixelUnshuffle(4)
        self.register_buffer("position_encoding", enc)

        # Figure out image projection (before concat with position encoding)
        if image_channels_before_concat is not None:
            self.image_project = nn.Linear(num_image_channels, image_channels_before_concat)
            num_image_channels = image_channels_before_concat
        else:
            self.image_project = nn.Identity()

        # Figure out output number of channels (image + position encoding)
        _num_output_channels = num_image_channels + num_position_encoding_channels
        if num_output_channels is not None:
            self.project = nn.Linear(_num_output_channels, num_output_channels)
            _num_output_channels = num_output_channels
        else:
            self.project = nn.Identity()

        self._num_output_channels = _num_output_channels

    @property
    def num_output_channels(self):
        return self._num_output_channels

    @property
    def num_position_encoding_channels(self, include_positions: bool = True) -> int:
        return len(self.spatial_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, x: Tensor) -> Tensor:
        x = self.space2depth(x)
        x = torch.moveaxis(x, 1, -1)  # Move BCHW to BHWC
        b, *d = x.shape

        if tuple(d) != self.image_shape_down:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape_down}")

        x = einops.rearrange(x, "b ... c -> b (...) c")
        x = self.image_project(x)

        x_enc = einops.repeat(self.position_encoding, "... -> b ...", b=b)
        x = torch.cat([x, x_enc], dim=-1)
        x = self.project(x)
        return x


if __name__ == "__main__":
    adapter = (
        FourierPEImageInputAdapter(
            image_shape=(512, 384, 3),
            num_frequency_bands=64,
            include_positions=True,
            image_channels_before_concat=256,
            num_output_channels=128,
        )
        .cuda()
        .eval()
    )

    print(count_parameters(adapter))
    input = torch.randn((4, 3, 512, 384)).cuda()

    y = adapter(input)
    print(describe_outputs(y))

    assert y.size(-1) == adapter.num_output_channels
