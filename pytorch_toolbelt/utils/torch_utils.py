"""Common functions to marshal data to/from PyTorch

"""
import collections
import warnings
from typing import Tuple

import numpy as np
import torch
from torch import nn


def set_trainable(module: nn.Module, trainable=True, freeze_bn=True):
    """
    Change 'requires_grad' value for module and it's child modules and
    optionally freeze batchnorm modules.
    :param module: Module to change
    :param trainable: True to enable training
    :param freeze_bn: True to freeze batch norm
    :return: None
    """
    trainable = bool(trainable)
    freeze_bn = bool(freeze_bn)

    for param in module.parameters():
        param.requires_grad = trainable

    # TODO: Add support for ABN, InplaceABN
    bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm

    if isinstance(module, bn_types):
        module.track_running_stats = freeze_bn

    for m in module.modules():
        if isinstance(m, bn_types):
            module.track_running_stats = freeze_bn


def freeze_bn(module: nn.Module):
    """Freezes BatchNorm
    """
    warnings.warn("This method is deprecated. Please use `set_trainable`.")
    set_trainable(module, True, False)


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
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
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

    raise ValueError('Unsupported input type' + str(type(x)))


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def tensor_from_mask_image(mask: np.ndarray) -> torch.Tensor:
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    return tensor_from_rgb_image(mask)


def rgb_image_from_tensor(
        image: torch.Tensor, mean, std, max_pixel_value=255.0, dtype=np.uint8
) -> np.ndarray:
    image = np.moveaxis(to_numpy(image), 0, -1)
    mean = to_numpy(mean)
    std = to_numpy(std)
    rgb_image = (max_pixel_value * (image * std + mean)).astype(dtype)
    return rgb_image


def maybe_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_optimizable_parameters(model: nn.Module):
    """
    Return list of optimizable parameters from the model
    :param model:
    :return:
    """
    return filter(lambda x: x.requires_grad, model.parameters())


def transfer_weights(model: nn.Module,
                     model_state_dict: collections.OrderedDict):
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
            model.load_state_dict(
                collections.OrderedDict([(name, value)]), strict=False
            )
        except Exception as e:
            print(e)
