"""Common functions to marshal data to/from PyTorch

"""
import collections
from typing import Optional, Sequence, Iterator, Union, Dict

import numpy as np
import torch
from torch import nn

__all__ = [
    "freeze_model",
    "rgb_image_from_tensor",
    "tensor_from_mask_image",
    "tensor_from_rgb_image",
    "count_parameters",
    "get_optimizable_parameters",
    "transfer_weights",
    "maybe_cuda",
    "mask_from_tensor",
    "logit",
    "to_numpy",
    "to_tensor",
]

from torch.nn import Parameter


def freeze_model(module: nn.Module, freeze_parameters: Optional[bool] = True, freeze_bn: Optional[bool] = True):
    """
    Change 'requires_grad' value for module and it's child modules and
    optionally freeze batchnorm modules.
    :param module: Module to change
    :param freeze_parameters: True to freeze parameters; False - to enable parameters optimization.
        If None - current state is not changed.
    :param freeze_bn: True to freeze batch norm; False - to enable BatchNorm updates.
        If None - current state is not changed.
    :return: None
    """
    bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm

    if freeze_parameters is not None:
        for param in module.parameters():
            param.requires_grad = not freeze_parameters

    if freeze_bn is not None:
        if isinstance(module, bn_types):
            module.track_running_stats = not freeze_bn

        for m in module.modules():
            if isinstance(m, bn_types):
                module.track_running_stats = not freeze_bn


def logit(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
    x = torch.clamp(x.float(), eps, 1.0 - eps)
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

    raise ValueError("Unsupported input type" + str(type(x)))


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.moveaxis(image, -1, 0)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    return image


def tensor_from_mask_image(mask: np.ndarray) -> torch.Tensor:
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    return tensor_from_rgb_image(mask)


def rgb_image_from_tensor(image: torch.Tensor, mean, std, max_pixel_value=255.0, dtype=np.uint8) -> np.ndarray:
    image = np.moveaxis(to_numpy(image), 0, -1)
    mean = to_numpy(mean)
    std = to_numpy(std)
    rgb_image = (max_pixel_value * (image * std + mean)).astype(dtype)
    return rgb_image


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


def get_optimizable_parameters(model: nn.Module) -> Iterator[Parameter]:
    """
    Return list of parameters with requires_grad=True from the model.
    This function allows easily get all parameters that should be optimized.
    :param model: An instance of nn.Module.
    :return: Parameters with requires_grad=True.
    """
    return filter(lambda x: x.requires_grad, model.parameters())


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
