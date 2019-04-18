from collections import Sized, Iterable

import torch
from torch import Tensor


def torch_none(x: Tensor):
    return x


def torch_rot90_(x: Tensor):
    return x.transpose_(2, 3).flip(2)


def torch_rot90(x: Tensor):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x: Tensor):
    return x.flip(2).flip(3)


def torch_rot270(x: Tensor):
    return x.transpose(2, 3).flip(3)


def torch_flipud(x: Tensor):
    return x.flip(2)


def torch_fliplp(x: Tensor):
    return x.flip(3)


def torch_transpose(x: Tensor):
    return x.transpose(2, 3)


def torch_transpose_(x: Tensor):
    return x.transpose_(2, 3)


def torch_transpose2(x: Tensor):
    return x.transpose(3, 2)


def pad_image_tensor(image_tensor: Tensor, pad_size: int = 32):
    """Pad input tensor to make it's height and width dividable by @pad_size

    :param image_tensor: Input tensor of shape NCHW
    :param pad_size: Pad size
    :return: Tuple of output tensor and pad params. Second argument can be used to reverse pad operation of model output
    """
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    if isinstance(pad_size, Sized) and isinstance(pad_size, Iterable) and len(pad_size) == 2:
        pad_height, pad_width = [int(val) for val in pad_size]
    elif isinstance(pad_size, int):
        pad_height = pad_width = pad_size
    else:
        raise ValueError(f"Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar.")

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


def unpad_tensor(image_tensor, pad):
    pad_left, pad_right, pad_top, pad_btm = pad
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    return image_tensor[..., pad_top:rows - pad_btm, pad_left: cols - pad_right]


def unpad_xyxy_bboxes(bboxes_tensor: torch.Tensor, pad, dim=-1):
    pad_left, pad_right, pad_top, pad_btm = pad
    pad = torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype).to(bboxes_tensor.device)

    if dim == -1:
        dim = len(bboxes_tensor.size()) - 1

    expand_dims = list(set(range(len(bboxes_tensor.size()))) - {dim})
    for i, dim in enumerate(expand_dims):
        pad = pad.unsqueeze(dim)

    return bboxes_tensor - pad
