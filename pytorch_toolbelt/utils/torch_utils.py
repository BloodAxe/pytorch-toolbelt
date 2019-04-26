"""Common functions to marshal data to/from PyTorch

"""

from typing import Tuple

import torch
import numpy as np
from torch import nn


def freeze_bn(module: nn.Module):
    """Freezes BatchNorm
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.track_running_stats = False

    for m in module.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False


def logit(x: torch.Tensor, eps=1e-5):
    x = torch.clamp(x.float(), eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count number of total and trainable parameters of a model
    :param model: A model
    :return: Tuple (total, trainable)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def to_numpy(x) -> np.ndarray:
    """
    Convert whatever to numpy array
    :param x: List, tuple, PyTorch tensor or numpy array
    :return: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise ValueError('Unsupported type')
    return x


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def tensor_from_mask_image(mask: np.ndarray) -> torch.Tensor:
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    return tensor_from_rgb_image(mask)


def rgb_image_from_tensor(image: torch.Tensor, mean, std, max_pixel_value=255.0) -> np.ndarray:
    image = np.moveaxis(to_numpy(image), 0, -1)
    mean = to_numpy(mean)
    std = to_numpy(std)
    rgb_image = (max_pixel_value * (image * std + mean) + 0.5).astype(np.uint8)
    return rgb_image


def maybe_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x
