"""Implementation of GPU-friendly test-time augmentation for image segmentation and classification tasks.

Despite this is called test-time augmentation, these method can be used at training time as well since all
transformation written in PyTorch and respect gradients flow.
"""
from functools import partial
from typing import Tuple, List, Optional, Union, Callable, Dict

import torch
from torch import Tensor, nn
from torch.nn.functional import interpolate
from ..utils.support import pytorch_toolbelt_deprecated
from . import functional as F

__all__ = [
    "MultiscaleTTAWrapper",
    "TTAWrapper",
    "GeneralizedTTA",
    "d2_image_augment",
    "d2_image_deaugment",
    "d4_image2label",
    "d4_image2mask",
    "d4_image_augment",
    "d4_image_deaugment",
    "ms_image_augment",
    "ms_image_deaugment",
    "fivecrop_image2label",
    "fliplr_image2label",
    "fliplr_image2mask",
    "fliplr_image_augment",
    "fliplr_image_deaugment",
    "flips_augment",
    "flips_deaugment",
    "tencrop_image2label",
]

MaybeStrOrCallable = Optional[Union[str, Callable]]


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


def fliplr_image_augment(image: Tensor) -> Tensor:
    """
    Augment input tensor using flip from left to right
    Args:
        image: Tensor of [B,C,H,W] shape

    Returns:
        Tensor of [B * 2, C, H, W] shape with:
            - Original tensor rotated by 180 degrees
            - Horisonalty-flipped tensor

    """
    return torch.cat([image, F.torch_fliplr(image)], dim=0)


def fliplr_image_deaugment(image: Tensor, reduction: MaybeStrOrCallable = "mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was fliplr-augmented image (See fliplr_image_augment).
    Args:
        image: Tensor of [B * 2, C, H, W] shape
        reduction: Reduction model for aggregating outputs. Default is taking mean.

    Returns:
        Tensor of [B, C, H, W] shape if reduction is not None or "none", otherwise returns de-augmented tensor of
        [2, B, C, H, W] shape
    """
    assert image.size(0) % 2 == 0

    b1, b2 = torch.chunk(image, 2)

    image: Tensor = torch.stack([b1, F.torch_fliplr(b2)])

    if reduction == "mean":
        image = image.mean(dim=0)
    if reduction == "sum":
        image = image.sum(dim=0)
    if callable(reduction):
        image = reduction(image, dim=0)
    return image


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
    return torch.cat(
        [
            image,
            F.torch_rot180(image),
            F.torch_fliplr(image),
            F.torch_flipud(image),
        ],
        dim=0,
    )


def d2_image_deaugment(image: Tensor, reduction: MaybeStrOrCallable = "mean") -> Tensor:
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
        [
            b1,
            F.torch_rot180(b2),
            F.torch_fliplr(b3),
            F.torch_flipud(b4),
        ]
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


def d4_image_deaugment(image: Tensor, reduction: MaybeStrOrCallable = "mean") -> Tensor:
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


def flips_deaugment(
    image: Tensor,
    reduction: MaybeStrOrCallable = "mean",
) -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was flip-augmented image (See flips_augment).
    Args:
        image: Tensor of [B * 3, C, H, W] shape
        reduction: If True performs averaging of 8 outputs, otherwise - summation.

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
    if callable(reduction):
        image = reduction(image, dim=0)
    return image


@pytorch_toolbelt_deprecated("This class is deprecated. Please use GeneralizedTTA instead")
class TTAWrapper(nn.Module):
    def __init__(self, model: nn.Module, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


def ms_image_augment(
    image: Tensor, size_offsets: List[Union[int, Tuple[int, int]]], mode="bilinear", align_corners=True
) -> List[Tensor]:
    """
    Multi-scale image augmentation. This function create list of resized tensors from the input one.
    """
    batch_size, channels, rows, cols = image.size()
    augmented_inputs = []
    for offset in size_offsets:
        # TODO: Add support of tuple (row_offset, col_offset)
        scale_size = rows + offset, cols + offset
        scaled_input = torch.nn.functional.interpolate(image, size=scale_size, mode=mode, align_corners=align_corners)
        augmented_inputs.append(scaled_input)
    return augmented_inputs


def ms_image_deaugment(
    images: List[Tensor],
    size_offsets: List[Union[int, Tuple[int, int]]],
    reduction: MaybeStrOrCallable = "mean",
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Tensor:
    if len(images) != len(size_offsets):
        raise ValueError("Number of images must be equal to number of size offsets")

    deaugmented_outputs = []
    for image, offset in zip(images, size_offsets):
        batch_size, channels, rows, cols = image.size()
        # TODO: Add support of tuple (row_offset, col_offset)
        original_size = rows - offset, cols - offset
        scaled_image = torch.nn.functional.interpolate(
            image, size=original_size, mode=mode, align_corners=align_corners
        )
        deaugmented_outputs.append(scaled_image)

    deaugmented_outputs = torch.stack(deaugmented_outputs)
    if reduction == "mean":
        deaugmented_outputs = deaugmented_outputs.mean(dim=0)
    if reduction == "sum":
        deaugmented_outputs = deaugmented_outputs.sum(dim=0)
    if callable(reduction):
        deaugmented_outputs = reduction(deaugmented_outputs, dim=0)

    return deaugmented_outputs


@pytorch_toolbelt_deprecated("This class is deprecated. Please use MultiscaleTTA instead")
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


class GeneralizedTTA(nn.Module):
    """
    Example:
        tta_model = GeneralizedTTA(model,
            augment_fn=tta.d2_image_augment,
            deaugment_fn={
                OUTPUT_MASK_KEY: tta.d2_image_deaugment,
                OUTPUT_EDGE_KEY: tta.d2_image_deaugment,
            },


    Notes:
        Input tensors must be square for D2/D4 or similar types of augmentation
    """

    def __init__(
        self,
        model: nn.Module,
        augment_fn: Union[Callable, Dict[str, Callable], List[Callable]],
        deaugment_fn: Union[Callable, Dict[str, Callable], List[Callable]],
    ):
        super().__init__()
        self.model = model
        self.augment_fn = augment_fn
        self.deaugment_fn = deaugment_fn

    def forward(self, *input, **kwargs):
        # Augment & forward
        if isinstance(self.augment_fn, dict):
            if len(input) != 0:
                raise ValueError("Input for GeneralizedTTA must be exactly one tensor")
            augmented_inputs = dict(
                (key, augment(value)) for (key, value), augment in zip(kwargs.items(), self.augment_fn)
            )
            outputs = self.model(**augmented_inputs)
        elif isinstance(self.augment_fn, (list, tuple)):
            if len(kwargs) != 0:
                raise ValueError("Input for GeneralizedTTA must be exactly one tensor")
            augmented_inputs = [augment(x) for x, augment in zip(input, self.augment_fn)]
            outputs = self.model(*augmented_inputs)
        else:
            if len(input) != 1:
                raise ValueError("Input for GeneralizedTTA must be exactly one tensor")
            if len(kwargs) != 0:
                raise ValueError("Input for GeneralizedTTA must be exactly one tensor")
            augmented_input = self.augment_fn(input[0])
            outputs = self.model(augmented_input)

        # Deaugment outputs
        if isinstance(self.deaugment_fn, dict):
            if not isinstance(outputs, dict):
                raise ValueError("Output of the model must be a dict")

            deaugmented_output = dict((key, self.deaugment_fn[key](value)) for (key, value) in outputs.items())
        elif isinstance(self.deaugment_fn, (list, tuple)):
            if not isinstance(outputs, (dict, tuple)):
                raise ValueError("Output of the model must be a dict")

            deaugmented_output = [deaugment(value) for value, deaugment in zip(outputs, self.deaugment_fn)]
        else:
            deaugmented_output = self.deaugment_fn(outputs)

        return deaugmented_output
