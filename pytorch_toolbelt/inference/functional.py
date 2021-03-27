from collections.abc import Sized, Iterable
from typing import Union, Tuple

import torch
from torch import Tensor

from ..utils.support import pytorch_toolbelt_deprecated

__all__ = [
    "torch_none",
    "torch_rot90",
    "torch_rot90_cw",
    "torch_rot90_ccw",
    "torch_transpose_rot90_cw",
    "torch_transpose_rot90_ccw",
    "torch_rot90_ccw_transpose",
    "torch_rot90_cw_transpose",
    "torch_rot180",
    "torch_rot270",
    "torch_fliplr",
    "torch_flipud",
    "torch_transpose",
    "torch_transpose2",
    "torch_transpose_",
    "pad_image_tensor",
    "unpad_image_tensor",
    "unpad_xyxy_bboxes",
    "geometric_mean",
    "harmonic_mean",
]


def torch_none(x: Tensor) -> Tensor:
    """
    Return input argument without any modifications
    :param x: input tensor
    :return: x
    """
    return x


def torch_rot90_ccw(x):
    return x.rot90(k=1, dims=(2, 3))


def torch_rot90_cw(x):
    return x.rot90(k=-1, dims=(2, 3))


def torch_rot90_ccw_transpose(x):
    return x.rot90(k=1, dims=(2, 3)).transpose(2, 3)


def torch_rot90_cw_transpose(x):
    return x.rot90(k=-1, dims=(2, 3)).transpose(2, 3)


def torch_transpose_rot90_ccw(x: Tensor):
    return x.transpose(2, 3).rot90(k=1, dims=(2, 3))


def torch_transpose_rot90_cw(x):
    return x.transpose(2, 3).rot90(k=-1, dims=(2, 3))


@pytorch_toolbelt_deprecated("Function torch_rot90 has been marked as deprecated. Please use torch_rot90_ccw instead")
def torch_rot90(x: Tensor):
    """
    Rotate 4D image tensor by 90 degrees
    :param x:
    :return:
    """
    return torch_rot90_ccw(x)


def torch_rot180(x: Tensor):
    """
    Rotate 4D image tensor by 180 degrees
    :param x:
    :return:
    """
    return torch.rot90(x, k=2, dims=(2, 3))


def torch_rot180_transpose(x):
    return x.rot90(k=2, dims=(2, 3)).transpose(2, 3)


def torch_transpose_rot180(x):
    return x.transpose(2, 3).rot90(k=2, dims=(2, 3))


@pytorch_toolbelt_deprecated("Function torch_rot270 has been marked as deprecated. Please use torch_rot90_cw instead")
def torch_rot270(x: Tensor):
    """
    Rotate 4D image tensor by 270 degrees
    :param x:
    :return:
    """
    return torch_rot90_cw(x)


def torch_flipud(x: Tensor):
    """
    Flip 4D image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x: Tensor):
    """
    Flip 4D image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)


def torch_transpose(x: Tensor):
    """
    Transpose 4D image tensor by main image diagonal
    :param x:
    :return:
    """
    return x.transpose(2, 3)


def torch_transpose_(x: Tensor):
    return x.transpose_(2, 3)


def torch_transpose2(x: Tensor):
    """
    Transpose 4D image tensor by second image diagonal
    :param x:
    :return:
    """
    return x.transpose(3, 2)


def pad_image_tensor(image_tensor: Tensor, pad_size: Union[int, Tuple[int, int]] = 32):
    """Pad input tensor to make it's height and width dividable by @pad_size

    :param image_tensor: 4D image tensor of shape NCHW
    :param pad_size: Pad size
    :return: Tuple of output tensor and pad params. Second argument can be used to reverse pad operation of model output
    """
    if len(image_tensor.size()) != 4:
        raise ValueError("Tensor must have rank 4 ([B,C,H,W])")

    rows, cols = image_tensor.size(2), image_tensor.size(3)
    if isinstance(pad_size, Sized) and isinstance(pad_size, Iterable) and len(pad_size) == 2:
        pad_height, pad_width = [int(val) for val in pad_size]
    elif isinstance(pad_size, int):
        pad_height = pad_width = pad_size
    else:
        raise ValueError(
            f"Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar."
        )

    if rows > pad_height:
        pad_rows = rows % pad_height
        pad_rows = pad_height - pad_rows if pad_rows > 0 else 0
    else:
        pad_rows = pad_height - rows

    if cols > pad_width:
        pad_cols = cols % pad_width
        pad_cols = pad_width - pad_cols if pad_cols > 0 else 0
    else:
        pad_cols = pad_width - cols

    if pad_rows == 0 and pad_cols == 0:
        return image_tensor, (0, 0, 0, 0)

    pad_top = pad_rows // 2
    pad_btm = pad_rows - pad_top

    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    pad = [pad_left, pad_right, pad_top, pad_btm]
    image_tensor = torch.nn.functional.pad(image_tensor, pad)
    return image_tensor, pad


def unpad_image_tensor(image_tensor: Tensor, pad) -> Tensor:
    if len(image_tensor.size()) != 4:
        raise ValueError("Tensor must have rank 4 ([B,C,H,W])")

    pad_left, pad_right, pad_top, pad_btm = pad
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    return image_tensor[..., pad_top : rows - pad_btm, pad_left : cols - pad_right]


def unpad_xyxy_bboxes(bboxes_tensor: torch.Tensor, pad, dim=-1):
    pad_left, pad_right, pad_top, pad_btm = pad
    pad = torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype).to(bboxes_tensor.device)

    if dim == -1:
        dim = len(bboxes_tensor.size()) - 1

    expand_dims = list(set(range(len(bboxes_tensor.size()))) - {dim})
    for i, dim in enumerate(expand_dims):
        pad = pad.unsqueeze(dim)

    return bboxes_tensor - pad


def geometric_mean(x: Tensor, dim: int) -> Tensor:
    """
    Compute geometric mean along given dimension.
    This implementation assume values are in range (0...1) (Probabilities)
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    return x.log().mean(dim=dim).exp()


def harmonic_mean(x: Tensor, dim: int, eps: float = 1e-6) -> Tensor:
    """
    Compute harmonic mean along given dimension.
    This implementation assume values are in range (0...1) (Probabilities)
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce

    Returns:
        Tensor
    """
    x = torch.reciprocal(x.clamp_min(eps))
    x = torch.mean(x, dim=dim)
    x = torch.reciprocal(x.clamp_min(eps))
    return x
