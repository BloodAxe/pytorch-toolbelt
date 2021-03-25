"""Implementation of GPU-friendly test-time augmentation for image segmentation and classification tasks.

Despite this is called test-time augmentation, these method can be used at training time as well since all
transformation written in PyTorch and respect gradients flow.
"""
from collections import defaultdict
from functools import partial
from typing import Tuple, List, Optional, Union, Callable, Dict

import torch
from torch import Tensor, nn

from . import functional as F
from ..utils.support import pytorch_toolbelt_deprecated

__all__ = [
    "GeneralizedTTA",
    "MultiscaleTTA",
    "d2_image_augment",
    "d2_image_deaugment",
    "d2_labels_deaugment",
    "d4_image2label",
    "d4_image2mask",
    "d4_image_augment",
    "d4_image_deaugment",
    "d4_labels_deaugment",
    "fivecrop_image2label",
    "fivecrop_image_augment",
    "fivecrop_label_deaugment",
    "fliplr_image2label",
    "fliplr_image2mask",
    "fliplr_image_augment",
    "fliplr_image_deaugment",
    "fliplr_labels_deaugment",
    "flips_image_augment",
    "flips_image_deaugment",
    "flips_labels_deaugment",
    "ms_image_augment",
    "ms_image_deaugment",
    "tencrop_image2label",
]

MaybeStrOrCallable = Optional[Union[str, Callable]]


def _deaugment_averaging(x: Tensor, reduction: MaybeStrOrCallable) -> Tensor:
    """
    Helper method to average predictions of TTA-ed model.
    This function assumes TTA dimension is 0, e.g [T, B, C, Ci, Cj, ..]
    Args:
        x: Input tensor of shape [T, B, ... ]
        reduction: Reduction mode ("sum", "mean", "gmean", "hmean", function, None)

    Returns:
        Tensor of shape [B, C, Ci, Cj, ..]
    """
    if reduction == "mean":
        x = x.mean(dim=0)
    elif reduction == "sum":
        x = x.sum(dim=0)
    elif reduction in {"gmean", "geometric_mean"}:
        x = F.geometric_mean(x, dim=0)
    elif reduction in {"hmean", "harmonic_mean"}:
        x = F.harmonic_mean(x, dim=0)
    elif callable(reduction):
        x = reduction(x, dim=0)
    elif reduction in {None, "None", "none"}:
        pass
    else:
        raise KeyError(f"Unsupported reduction mode {reduction}")

    return x


def fivecrop_image_augment(image: Tensor, crop_size: Tuple[int, int]) -> Tensor:
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them.

    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    image_height, image_width = int(image.size(2)), int(image.size(3))
    crop_height, crop_width = crop_size

    if crop_height > image_height:
        raise ValueError(f"Tensor height ({image_height}) is less than requested crop size ({crop_height})")
    if crop_width > image_width:
        raise ValueError(f"Tensor width ({image_width}) is less than requested crop size ({crop_width})")

    bottom_crop_start = image_height - crop_height
    right_crop_start = image_width - crop_width
    crop_tl = image[..., :crop_height, :crop_width]
    crop_tr = image[..., :crop_height, right_crop_start:]
    crop_bl = image[..., bottom_crop_start:, :crop_width]
    crop_br = image[..., bottom_crop_start:, right_crop_start:]

    center_crop_y = (image_height - crop_height) // 2
    center_crop_x = (image_width - crop_width) // 2
    crop_cc = image[..., center_crop_y : center_crop_y + crop_height, center_crop_x : center_crop_x + crop_width]

    return torch.cat([crop_tl, crop_tr, crop_bl, crop_br, crop_cc], dim=0,)


def fivecrop_label_deaugment(logits: Tensor, reduction: MaybeStrOrCallable = "mean") -> Tensor:
    crop_tl, crop_tr, crop_bl, crop_br, crop_cc = torch.chunk(logits, 5)

    logits: Tensor = torch.stack([crop_tl, crop_tr, crop_bl, crop_br, crop_cc])
    return _deaugment_averaging(logits, reduction=reduction)


def fivecrop_image2label(model: nn.Module, image: Tensor, crop_size: Tuple) -> Tensor:
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them.

    :param model: Classification model
    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    input_aug = fivecrop_image_augment(image, crop_size)
    preds_aug = model(input_aug)
    output = fivecrop_label_deaugment(preds_aug)
    return output


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


def fliplr_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    for input image and horizontally flipped one.

    :param model:
    :param image:
    :return:
    """
    return fliplr_labels_deaugment(model(fliplr_image_augment(image)))


def fliplr_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and vertically flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    return fliplr_image_deaugment(model(fliplr_image_augment(image)))


def d4_image2label(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    return d4_labels_deaugment(model(d4_image_augment(image)))


def d4_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    of all D4 augmentations applied to input image.

    For segmentation we need to reverse the augmentation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    return d4_image_deaugment(model(d4_image_augment(image)))


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

    return _deaugment_averaging(image, reduction=reduction)


def d2_image_augment(image: Tensor) -> Tensor:
    """
    Augment input tensor using D2 symmetry group
    Args:
        image: Tensor of [B,C,H,W] shape

    Returns:
        Tensor of [B * 8, C, H, W] shape with:
            - Original tensor
            - Original tensor rotated by 180 degrees
            - Horizontally-flipped tensor
            - Vertically-flipped tensor

    """
    return torch.cat([image, F.torch_rot180(image), F.torch_fliplr(image), F.torch_flipud(image),], dim=0,)


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
        [b1, F.torch_rot180(b2), F.torch_fliplr(b3), F.torch_flipud(b4),]
    )

    return _deaugment_averaging(image, reduction=reduction)


def d2_labels_deaugment(logits: Tensor, reduction: MaybeStrOrCallable = "mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the is 2D tensor (See d2_augment).
    Args:
        logits: Tensor of [B * 4, C] shape
        reduction: Reduction model for aggregating outputs. Default is taking mean.

    Returns:
        Tensor of [B, C] shape if reduction is not None or "none", otherwise returns de-augmented tensor of
        [4, B, C] shape
    """
    assert logits.size(0) % 4 == 0

    b1, b2, b3, b4 = torch.chunk(logits, 4)
    logits: Tensor = torch.stack([b1, b2, b3, b4])

    return _deaugment_averaging(logits, reduction=reduction)


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


def d4_labels_deaugment(image: Tensor, reduction: MaybeStrOrCallable = "mean") -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the is 2D tensor (See d2_augment).
    Args:
        image: Tensor of [B * 8, C] shape
        reduction: Reduction model for aggregating outputs. Default is taking mean.

    Returns:
        Tensor of [B, C] shape if reduction is not None or "none", otherwise returns de-augmented tensor of
        [8, B, C] shape
    """
    if image.size(0) % 8 != 0:
        raise RuntimeError("Batch size must be divisable by 8")

    b1, b2, b3, b4, b5, b6, b7, b8 = torch.chunk(image, 8)
    image: Tensor = torch.stack([b1, b2, b3, b4, b5, b7, b7, b8])

    return _deaugment_averaging(image, reduction=reduction)


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
    if image.size(0) % 8 != 0:
        raise RuntimeError("Batch size must be divisable by 8")

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
    return _deaugment_averaging(image, reduction=reduction)


def flips_image_augment(image: Tensor) -> Tensor:
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


def flips_image_deaugment(image: Tensor, reduction: MaybeStrOrCallable = "mean",) -> Tensor:
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
    return _deaugment_averaging(image, reduction=reduction)


def fliplr_labels_deaugment(logits: Tensor, reduction: MaybeStrOrCallable = "mean",) -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was fliplr-augmented image (See fliplr_image_augment).
    Args:
        logits: Tensor of [B * 2, C] shape
        reduction: If True performs averaging of 3 outputs, otherwise - summation.

    Returns:
        Tensor of [B, C, H, W] shape.
    """
    if logits.size(0) % 2 != 0:
        raise RuntimeError("Batch size must be divisable by 2")

    orig, flipped_lr = torch.chunk(logits, 2)
    logits: Tensor = torch.stack([orig, flipped_lr])
    return _deaugment_averaging(logits, reduction=reduction)


def flips_labels_deaugment(logits: Tensor, reduction: MaybeStrOrCallable = "mean",) -> Tensor:
    """
    Deaugment input tensor (output of the model) assuming the input was flip-augmented image (See flips_image_augment).
    Args:
        logits: Tensor of [B * 3, C] shape
        reduction: If True performs averaging of 3 outputs, otherwise - summation.

    Returns:
        Tensor of [B, C, H, W] shape.
    """
    if logits.size(0) % 3 != 0:
        raise RuntimeError("Batch size must be divisable by 3")

    orig, flipped_lr, flipped_ud = torch.chunk(logits, 3)
    logits: Tensor = torch.stack([orig, flipped_lr, flipped_ud])
    return _deaugment_averaging(logits, reduction=reduction)


@pytorch_toolbelt_deprecated("This class is deprecated. Please use GeneralizedTTA instead")
class TTAWrapper(nn.Module):
    def __init__(self, model: nn.Module, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)


def ms_image_augment(
    image: Tensor, size_offsets: List[Union[int, Tuple[int, int]]], mode="bilinear", align_corners=False
) -> List[Tensor]:
    """
    Multi-scale image augmentation. This function create list of resized tensors from the input one.
    """
    batch_size, channels, rows, cols = image.size()
    augmented_inputs = []
    for offset in size_offsets:
        if isinstance(offset, (tuple, list)):
            rows_offset, cols_offset = offset
        else:
            rows_offset, cols_offset = offset, offset

        if rows_offset == 0 and cols_offset == 0:
            augmented_inputs.append(image)
        else:
            scale_size = rows + rows_offset, cols + cols_offset
            scaled_input = torch.nn.functional.interpolate(
                image, size=scale_size, mode=mode, align_corners=align_corners
            )
            augmented_inputs.append(scaled_input)
    return augmented_inputs


def ms_labels_deaugment(
    logits: List[Tensor], size_offsets: List[Union[int, Tuple[int, int]]], reduction: MaybeStrOrCallable = "mean",
):
    """
    Deaugment logits

    Args:
        logits: List of tensors of shape [B, C]
        size_offsets:
        reduction:

    Returns:

    """
    if len(logits) != len(size_offsets):
        raise ValueError("Number of images must be equal to number of size offsets")

    logits = torch.stack(logits)
    return _deaugment_averaging(logits, reduction=reduction)


def ms_image_deaugment(
    images: List[Tensor],
    size_offsets: List[Union[int, Tuple[int, int]]],
    reduction: MaybeStrOrCallable = "mean",
    mode: str = "bilinear",
    align_corners: bool = True,
    stride: int = 1,
) -> Tensor:
    """

    Args:
        images: List of tensors of shape [B, C, Hi, Wi], [B, C, Hj, Wj], [B, C, Hk, Wk]
        size_offsets:
        reduction:
        mode:
        align_corners:
        stride: Stride of the output feature map w.r.t to model input size.
        Used to correctly scale size_offsets to match with size of output feature maps

    Returns:

    """
    if len(images) != len(size_offsets):
        raise ValueError("Number of images must be equal to number of size offsets")

    deaugmented_outputs = []
    for feature_map, offset in zip(images, size_offsets):
        if isinstance(offset, (tuple, list)):
            rows_offset, cols_offset = offset
        else:
            rows_offset, cols_offset = offset, offset

        if rows_offset == 0 and cols_offset == 0:
            deaugmented_outputs.append(feature_map)
        else:
            batch_size, channels, rows, cols = feature_map.size()
            original_size = rows - rows_offset // stride, cols - cols_offset // stride
            scaled_image = torch.nn.functional.interpolate(
                feature_map, size=original_size, mode=mode, align_corners=align_corners
            )
            deaugmented_outputs.append(scaled_image)

    deaugmented_outputs = torch.stack(deaugmented_outputs)
    return _deaugment_averaging(deaugmented_outputs, reduction=reduction)


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
        model: Union[nn.Module, nn.DataParallel],
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

            deaugmented_output = dict((key, self.deaugment_fn[key](outputs[key])) for key in self.deaugment_fn.keys())
        elif isinstance(self.deaugment_fn, (list, tuple)):
            if not isinstance(outputs, (dict, tuple)):
                raise ValueError("Output of the model must be a dict")

            deaugmented_output = [deaugment(value) for value, deaugment in zip(outputs, self.deaugment_fn)]
        else:
            deaugmented_output = self.deaugment_fn(outputs)

        return deaugmented_output


class MultiscaleTTA(nn.Module):
    def __init__(self, model: nn.Module, size_offsets: List[int], deaugment_fn: Optional[Dict[str, Callable]] = None):
        if deaugment_fn is None:
            deaugment_fn = defaultdict(lambda: ms_image_deaugment)
            self.keys = None
        else:
            self.keys = set(deaugment_fn.keys())

        super().__init__()
        self.model = model
        self.size_offsets = size_offsets
        self.deaugment_fn = deaugment_fn

    def forward(self, x):
        ms_inputs = ms_image_augment(x, size_offsets=self.size_offsets)
        ms_outputs = [self.model(x) for x in ms_inputs]

        outputs = {}
        if self.keys is None:
            keys = ms_outputs[0].keys()
        else:
            keys = self.keys

        for key in keys:
            deaugment_fn: Callable = self.deaugment_fn[key]
            values = [x[key] for x in ms_outputs]
            outputs[key] = deaugment_fn(values, self.size_offsets)

        return outputs
