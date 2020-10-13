"""Implementation of GPU-friendly test-time augmentation for image segmentation and classification tasks.

Despite this is called test-time augmentation, these method can be used at training time as well since all
transformation written in PyTorch and respect gradients flow.
"""
from functools import partial
from typing import Tuple, List, Optional, Union, Callable

import torch
from torch import Tensor, nn
from torch.nn.functional import interpolate

from . import functional as F

__all__ = [
    "MultiscaleTTAWrapper",
    "TTAWrapper",
    "d4_centernet_offset_deaugment",
    "d4_centernet_size_deaugment",
    "d4_image2label",
    "d4_image2mask",
    "d2_image_augment",
    "d2_image_deaugment",
    "d4_image_augment",
    "d4_image_deaugment",
    "fivecrop_image2label",
    "fliplr_image2label",
    "fliplr_image2mask",
    "flips_augment",
    "flips_deaugment",
    "tencrop_image2label",
]


def fliplr_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    for input image and horizontally flipped one.

    :param model:
    :param image:
    :return:
    """
    output = model(image) + model(F.torch_fliplr(image))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def fivecrop_image2label(model: nn.Module, image: Tensor, crop_size: Tuple) -> Tensor:
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them.

    :param model: Classification model
    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    image_height, image_width = int(image.size(2)), int(image.size(3))
    crop_height, crop_width = crop_size

    assert crop_height <= image_height
    assert crop_width <= image_width

    bottom_crop_start = image_height - crop_height
    right_crop_start = image_width - crop_width
    crop_tl = image[..., :crop_height, :crop_width]
    crop_tr = image[..., :crop_height, right_crop_start:]
    crop_bl = image[..., bottom_crop_start:, :crop_width]
    crop_br = image[..., bottom_crop_start:, right_crop_start:]

    assert crop_tl.size(2) == crop_height
    assert crop_tr.size(2) == crop_height
    assert crop_bl.size(2) == crop_height
    assert crop_br.size(2) == crop_height

    assert crop_tl.size(3) == crop_width
    assert crop_tr.size(3) == crop_width
    assert crop_bl.size(3) == crop_width
    assert crop_br.size(3) == crop_width

    center_crop_y = (image_height - crop_height) // 2
    center_crop_x = (image_width - crop_width) // 2

    crop_cc = image[..., center_crop_y : center_crop_y + crop_height, center_crop_x : center_crop_x + crop_width]
    assert crop_cc.size(2) == crop_height
    assert crop_cc.size(3) == crop_width

    output = model(crop_tl) + model(crop_tr) + model(crop_bl) + model(crop_br) + model(crop_cc)
    one_over_5 = float(1.0 / 5.0)
    return output * one_over_5


def tencrop_image2label(model: nn.Module, image: Tensor, crop_size: Tuple) -> Tensor:
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them and from their horisontally-flipped versions (10-Crop TTA).

    :param model: Classification model
    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    image_height, image_width = int(image.size(2)), int(image.size(3))
    crop_height, crop_width = crop_size

    assert crop_height <= image_height
    assert crop_width <= image_width

    bottom_crop_start = image_height - crop_height
    right_crop_start = image_width - crop_width
    crop_tl = image[..., :crop_height, :crop_width]
    crop_tr = image[..., :crop_height, right_crop_start:]
    crop_bl = image[..., bottom_crop_start:, :crop_width]
    crop_br = image[..., bottom_crop_start:, right_crop_start:]

    assert crop_tl.size(2) == crop_height
    assert crop_tr.size(2) == crop_height
    assert crop_bl.size(2) == crop_height
    assert crop_br.size(2) == crop_height

    assert crop_tl.size(3) == crop_width
    assert crop_tr.size(3) == crop_width
    assert crop_bl.size(3) == crop_width
    assert crop_br.size(3) == crop_width

    center_crop_y = (image_height - crop_height) // 2
    center_crop_x = (image_width - crop_width) // 2

    crop_cc = image[..., center_crop_y : center_crop_y + crop_height, center_crop_x : center_crop_x + crop_width]
    assert crop_cc.size(2) == crop_height
    assert crop_cc.size(3) == crop_width

    output = (
        model(crop_tl)
        + model(F.torch_fliplr(crop_tl))
        + model(crop_tr)
        + model(F.torch_fliplr(crop_tr))
        + model(crop_bl)
        + model(F.torch_fliplr(crop_bl))
        + model(crop_br)
        + model(F.torch_fliplr(crop_br))
        + model(crop_cc)
        + model(F.torch_fliplr(crop_cc))
    )

    one_over_10 = float(1.0 / 10.0)
    return output * one_over_10


def fliplr_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and vertically flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image) + F.torch_fliplr(model(F.torch_fliplr(image)))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def d4_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug in [F.torch_rot90, F.torch_rot180, F.torch_rot270]:
        x = model(aug(image))
        output = output + x

    image = F.torch_transpose(image)

    for aug in [F.torch_none, F.torch_rot90, F.torch_rot180, F.torch_rot270]:
        x = model(aug(image))
        output = output + x

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8


def d4_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    of all D4 augmentations applied to input image.

    For segmentation we need to reverse the augmentation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug, deaug in zip(
        [F.torch_rot90, F.torch_rot180, F.torch_rot270], [F.torch_rot270, F.torch_rot180, F.torch_rot90]
    ):
        x = deaug(model(aug(image)))
        output += x

    image = F.torch_transpose(image)

    for aug, deaug in zip(
        [F.torch_none, F.torch_rot90, F.torch_rot180, F.torch_rot270],
        [F.torch_none, F.torch_rot270, F.torch_rot180, F.torch_rot90],
    ):
        x = deaug(model(aug(image)))
        output += F.torch_transpose(x)

    one_over_8 = float(1.0 / 8.0)
    output *= one_over_8
    return output


def d2_image_augment(image: Tensor) -> Tensor:
    """
    Augment input tensor using D2 symmetry group
    Args:
        image: Tensor of [B,C,H,W] shape

    Returns:
        Tensor of [B * 8, C, H, W] shape with:
            - Original tensor
            - Original tensor rotated by 180 degrees
            - Horisonalty-flipped tensor
            - Vertically-flipped tensor

    """
    return torch.cat([image, F.torch_rot180(image), F.torch_fliplr(image), F.torch_flipud(image),], dim=0,)


def d2_image_deaugment(image: Tensor, reduction: Union[str, Callable] = "mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was D2-augmented image (See d2_augment).
    Args:
        image: Tensor of [B * 4, C, H, W] shape
        reduction: Reduction model for aggregating outputs. Default is taking mean.

    Returns:
        Tensor of [B, C, H, W] shape if reduction is not None or "none", otherwise returns de-augmented tensor of
        [4, B, C, H, W] shape
    """
    assert image.size(0) % 4 == 0

    b1, b2, b3, b4 = torch.chunk(image, 4)

    image: Tensor = torch.stack(
        [b1, F.torch_rot180(b2), F.torch_fliplr(b3), F.torch_flipud(b4),]
    )

    if reduction == "mean":
        image = image.mean(dim=0)
    if reduction == "sum":
        image = image.sum(dim=0)
    if callable(reduction):
        image = reduction(image, dim=0)
    return image


def d4_image_augment(image: Tensor) -> Tensor:
    """
    Augment input tensor using D4 symmetry group
    Args:
        image: Tensor of [B,C,H,W] shape

    Returns:
        Tensor of [B * 8, C, H, W] shape with:
            - Original tensor
            - Original tensor rotated by 90 degrees
            - Original tensor rotated by 180 degrees
            - Original tensor rotated by 180 degrees
            - Transposed tensor
            - Transposed tensor rotated by 90 degrees
            - Transposed tensor rotated by 180 degrees
            - Transposed tensor rotated by 180 degrees

    """
    if image.size(2) != image.size(3):
        raise ValueError(
            f"Input tensor must have number of rows equal to number of cols. "
            f"Got input tensor of shape {image.size()}"
        )
    image_t = F.torch_transpose(image)
    return torch.cat(
        [
            image,
            F.torch_rot90_cw(image),
            F.torch_rot180(image),
            F.torch_rot90_ccw(image),
            image_t,
            F.torch_rot90_cw(image_t),
            F.torch_rot180(image_t),
            F.torch_rot90_ccw(image_t),
        ],
        dim=0,
    )


def d4_image_deaugment(image: Tensor, reduction: Union[str, Callable] = "mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was D4-augmented image (See d4_augment).
    Args:
        image: Tensor of [B * 8, C, H, W] shape
        average: If True performs averaging of 8 outputs, otherwise - summation.

    Returns:
        Tensor of [B, C, H, W] shape if reduction is not None or "none", otherwise returns de-augmented tensor of
        [4, B, C, H, W] shape

    """
    assert image.size(0) % 8 == 0

    b1, b2, b3, b4, b5, b6, b7, b8 = torch.chunk(image, 8)

    image: Tensor = torch.stack(
        [
            b1,
            F.torch_rot90_ccw(b2),
            F.torch_rot180(b3),
            F.torch_rot90_cw(b4),
            F.torch_transpose(b5),
            F.torch_rot90_ccw_transpose(b6),
            F.torch_rot180_transpose(b7),
            F.torch_rot90_cw_transpose(b8),
        ]
    )

    if reduction == "mean":
        image = image.mean(dim=0)
    if reduction == "sum":
        image = image.sum(dim=0)
    if callable(reduction):
        image = reduction(image, dim=0)
    return image


def d4_centernet_size_deaugment(image: Tensor, reduction: Optional[str] = "mean") -> Tensor:
    """
    Deaugment input tensor width & height regression (for centernet) assuming the input was D4-augmented image (See d4_augment).
    Args:
        image: Tensor of [B * 8, 2, H, W] shape
        average: If True performs averaging of 8 outputs, otherwise - summation.

    Returns:
        Tensor of [B, C, H, W] shape.
    """
    assert image.size(1) == 2
    assert image.size(0) % 8 == 0

    b1, b2, b3, b4, b5, b6, b7, b8 = torch.chunk(image, 8)

    image: Tensor = torch.stack(
        [
            b1,
            F.torch_rot90_ccw(b2),
            F.torch_rot180(b3),
            F.torch_rot90_cw(b4),
            F.torch_transpose(b5),
            F.torch_rot90_ccw_transpose(b6),
            F.torch_rot180_transpose(b7),
            F.torch_rot90_cw_transpose(b8),
        ]
    )

    if reduction == "mean":
        image = image.mean(dim=0)
    if reduction == "sum":
        image = image.sum(dim=0)
    return image


def d4_centernet_offset_deaugment(image: Tensor, reduction: Optional[str] = "mean") -> Tensor:
    """
    Deaugment input tensor width & height offset (for centernet) assuming the input was D4-augmented image (See d4_augment).
    Args:
        image: Tensor of [B * 8, 2, H, W] shape
        average: If True performs averaging of 8 outputs, otherwise - summation.

    Returns:
        Tensor of [B, C, H, W] shape.
    """
    assert image.size(1) == 2
    assert image.size(0) % 8 == 0

    b1, b2, b3, b4, b5, b6, b7, b8 = torch.chunk(image, 8)

    image: Tensor = torch.stack(
        [
            b1,
            F.torch_rot90_ccw(b2),
            F.torch_rot180(b3),
            F.torch_rot90_cw(b4),
            F.torch_transpose(b5),
            F.torch_rot90_ccw_transpose(b6),
            F.torch_rot180_transpose(b7),
            F.torch_rot90_cw_transpose(b8),
        ]
    )

    if reduction == "mean":
        image = image.mean(dim=0)
    if reduction == "sum":
        image = image.sum(dim=0)
    return image


def cnet_swap_width_height(x: Tensor) -> Tensor:
    """
    For size and offset tensor of shape [B,2,H,W] change order of 0 and 1 channels in dimension 1
    """
    assert x.size(1) == 2
    return torch.cat([x[:, 1:2, ...], x[:, 0:1, ...]], dim=1)


def cnet_fliplr_offset(x: Tensor) -> Tensor:
    return torch.cat([1 - x[:, 0:1, ...], x[:, 1:2, ...]], dim=1)


def cnet_flipud_offset(x: Tensor) -> Tensor:
    return torch.cat([x[:, 0:1, ...], 1 - x[:, 1:2, ...]], dim=1)


def cnet_flip_offset(x: Tensor) -> Tensor:
    return 1 - x


def flips_augment(image: Tensor) -> Tensor:
    """
    Augment input tensor by adding vertically and horizontally flipped images to it.

    Args:
        image: Tensor of [B,C,H,W] shape

    Returns:
        Tensor of [B * 3, C, H, W] shape with:
            - Original tensor
            - Horizontally-flipped tensor
            - Vertically-flipped

    """
    return torch.cat([image, F.torch_fliplr(image), F.torch_flipud(image)], dim=0)


def flips_deaugment(image: Tensor, reduction: Optional[str] = "mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was flip-augmented image (See flips_augment).
    Args:
        image: Tensor of [B * 3, C, H, W] shape
        average: If True performs averaging of 8 outputs, otherwise - summation.

    Returns:
        Tensor of [B, C, H, W] shape.
    """
    batch_size: int = image.shape[0] // 3
    image: Tensor = torch.stack(
        [
            image[batch_size * 0 : batch_size * 1],
            F.torch_fliplr(image[batch_size * 1 : batch_size * 2]),
            F.torch_flipud(image[batch_size * 2 : batch_size * 3]),
        ]
    )

    if reduction == "mean":
        image = image.mean(dim=0)
    if reduction == "sum":
        image = image.sum(dim=0)
    return image


class TTAWrapper(nn.Module):
    def __init__(self, model: nn.Module, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


class MultiscaleTTAWrapper(nn.Module):
    """
    Multiscale TTA wrapper module
    """

    def __init__(self, model: nn.Module, scale_levels: List[float] = None, size_offsets: List[int] = None):
        """
        Initialize multi-scale TTA wrapper

        :param model: Base model for inference
        :param scale_levels: List of additional scale levels,
            e.g: [0.5, 0.75, 1.25]
        """
        super().__init__()
        assert scale_levels or size_offsets, "Either scale_levels or size_offsets must be set"
        assert not (scale_levels and size_offsets), "Either scale_levels or size_offsets must be set"
        self.model = model
        self.scale_levels = scale_levels
        self.size_offsets = size_offsets

    def forward(self, input: Tensor) -> Tensor:
        h = input.size(2)
        w = input.size(3)

        out_size = h, w
        output = self.model(input)

        if self.scale_levels:
            for scale in self.scale_levels:
                dst_size = int(h * scale), int(w * scale)
                input_scaled = interpolate(input, dst_size, mode="bilinear", align_corners=False)
                output_scaled = self.model(input_scaled)
                output_scaled = interpolate(output_scaled, out_size, mode="bilinear", align_corners=False)
                output += output_scaled
            output /= 1.0 + len(self.scale_levels)
        elif self.size_offsets:
            for offset in self.size_offsets:
                dst_size = int(h + offset), int(w + offset)
                input_scaled = interpolate(input, dst_size, mode="bilinear", align_corners=False)
                output_scaled = self.model(input_scaled)
                output_scaled = interpolate(output_scaled, out_size, mode="bilinear", align_corners=False)
                output += output_scaled
            output /= 1.0 + len(self.size_offsets)

        return output
