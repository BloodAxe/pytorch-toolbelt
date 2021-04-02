"""Common functions to marshal data to/from PyTorch

"""
import collections
from typing import Optional, Sequence, Union, Dict, List, Any

import numpy as np
import torch
from torch import nn, Tensor
from .support import pytorch_toolbelt_deprecated

__all__ = [
    "count_parameters",
    "image_to_tensor",
    "logit",
    "mask_from_tensor",
    "maybe_cuda",
    "rgb_image_from_tensor",
    "tensor_from_mask_image",
    "tensor_from_rgb_image",
    "to_numpy",
    "to_tensor",
    "resize_like",
    "resize_as",
    "transfer_weights",
    "softmax_over_dim_0",
    "softmax_over_dim_1",
    "softmax_over_dim_2",
    "softmax_over_dim_3",
    "argmax_over_dim_0",
    "argmax_over_dim_1",
    "argmax_over_dim_2",
    "argmax_over_dim_3",
    "sigmoid_with_threshold",
]


def softmax_over_dim_0(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=0)


def softmax_over_dim_1(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=1)


def softmax_over_dim_2(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=2)


def softmax_over_dim_3(x: Tensor) -> Tensor:
    return torch.softmax(x, dim=3)


def argmax_over_dim_0(x: Tensor) -> Tensor:
    return torch.argmax(x, dim=0)


def argmax_over_dim_1(x: Tensor) -> Tensor:
    return torch.argmax(x, dim=1)


def argmax_over_dim_2(x: Tensor) -> Tensor:
    return torch.argmax(x, dim=2)


def argmax_over_dim_3(x: Tensor) -> Tensor:
    return torch.argmax(x, dim=3)


def sigmoid_with_threshold(x: Tensor, threshold):
    return x.float().sigmoid().gt(threshold)


def logit(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    Compute inverse of sigmoid of the input.
    Note: This function has not been tested for numerical stability.
    :param x:
    :param eps:
    :return:
    """
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def count_parameters(model: nn.Module, keys: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """
    Count number of total and trainable parameters of a model
    :param model: A model
    :param keys: Optional list of top-level blocks
    :return: Tuple (total, trainable)
    """
    if keys is None:
        keys = ["encoder", "decoder", "logits", "head", "final"]
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    parameters = {"total": total, "trainable": trainable}

    for key in keys:
        if hasattr(model, key) and model.__getattr__(key) is not None:
            parameters[key] = int(sum(p.numel() for p in model.__getattr__(key).parameters()))

    return parameters


def to_numpy(x: Union[torch.Tensor, np.ndarray, Any]) -> np.ndarray:
    """
    Convert whatever to numpy array

    Args:
        :param x: List, tuple, PyTorch tensor or numpy array

    Returns:
        :return: Numpy array
    """
    if torch.is_tensor(x):
        return x.data.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {"O", "M", "U", "S"}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

    raise ValueError("Unsupported input type" + str(type(x)))


def image_to_tensor(image: np.ndarray, dummy_channels_dim=True) -> torch.Tensor:
    """
    Convert numpy image (RGB, BGR, Grayscale, SAR, Mask image, etc.) to tensor

    Args:
        image: A numpy array of [H,W,C] shape
        dummy_channels_dim: If True, and image has [H,W] shape adds dummy channel, so that
            output tensor has shape [1, H, W]

    See also:
        rgb_image_from_tensor - To convert tensor image back to RGB with denormalization
        mask_from_tensor

    Returns:
        Torch tensor of [C,H,W] or [H,W] shape (dummy_channels_dim=False).
    """
    if len(image.shape) not in {2, 3}:
        raise ValueError(f"Image must have shape [H,W] or [H,W,C]. Got image with shape {image.shape}")

    if len(image.shape) == 2:
        if dummy_channels_dim:
            image = np.expand_dims(image, 0)
    else:
        # HWC -> CHW
        image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


@pytorch_toolbelt_deprecated("This function is deprecated, please use image_to_tensor instead")
def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    return image_to_tensor(image)


@pytorch_toolbelt_deprecated("This function is deprecated, please use image_to_tensor instead")
def tensor_from_mask_image(mask: np.ndarray) -> torch.Tensor:
    return image_to_tensor(mask)


def rgb_image_from_tensor(
    image: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), min_pixel_value=0.0, max_pixel_value=255.0, dtype=np.uint8
) -> np.ndarray:
    """
    Convert numpy image (RGB, BGR, Grayscale, SAR, Mask image, etc.) to tensor

    Args:
        image: A torch tensor of [C,H,W] shape
    """
    image = np.moveaxis(to_numpy(image), 0, -1)
    mean = to_numpy(mean)
    std = to_numpy(std)
    rgb_image = (max_pixel_value * (image * std + mean))
    rgb_image = np.clip(rgb_image, a_min=min_pixel_value, a_max=max_pixel_value)
    return rgb_image.astype(dtype)


def mask_from_tensor(mask: torch.Tensor, squeeze_single_channel=False, dtype=None) -> np.ndarray:
    mask = np.moveaxis(to_numpy(mask), 0, -1)
    if squeeze_single_channel and mask.shape[-1] == 1:
        mask = np.squeeze(mask, -1)

    if dtype is not None:
        mask = mask.astype(dtype)
    return mask


def maybe_cuda(x: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
    """
    Move input Tensor or Module to CUDA device if CUDA is available.
    :param x:
    :return:
    """
    if torch.cuda.is_available():
        return x.cuda()
    return x


def transfer_weights(model: nn.Module, model_state_dict: collections.OrderedDict):
    """
    Copy weights from state dict to model, skipping layers that are incompatible.
    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.
    :param model: Model to load weights into
    :param model_state_dict: Model state dict to load weights from
    :return: None
    """
    for name, value in model_state_dict.items():
        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
        except Exception as e:
            print(e)


def resize_like(x: Tensor, target: Tensor, mode="bilinear", align_corners=True) -> Tensor:
    """
    Resize input tensor to have the same spatial dimensions as target
    Args:
        x:
        target:
        mode:
        align_corners:

    Returns:

    """
    return torch.nn.functional.interpolate(x, target.size()[2:], mode=mode, align_corners=align_corners)


resize_as = resize_like
