"""Common functions to marshal data to/from PyTorch

"""

import collections
import dataclasses
import functools
import warnings
import logging
from typing import Optional, Sequence, Union, Dict, List, Any, Iterable, Callable, Tuple, Mapping

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

from .support import pytorch_toolbelt_deprecated

logger = logging.getLogger("pytorch_toolbelt.utils")

__all__ = [
    "argmax_over_dim_0",
    "argmax_over_dim_1",
    "argmax_over_dim_2",
    "argmax_over_dim_3",
    "count_parameters",
    "image_to_tensor",
    "int_to_string_human_friendly",
    "logit",
    "mask_from_tensor",
    "maybe_cuda",
    "resize_as",
    "resize_like",
    "rgb_image_from_tensor",
    "sigmoid_with_threshold",
    "softmax_over_dim_0",
    "softmax_over_dim_1",
    "softmax_over_dim_2",
    "softmax_over_dim_3",
    "tensor_from_mask_image",
    "tensor_from_rgb_image",
    "to_numpy",
    "to_tensor",
    "transfer_weights",
    "move_to_device",
    "move_to_device_non_blocking",
    "describe_outputs",
    "get_collate_for_dataset",
    "get_non_wrapped_model",
    "container_to_tensor",
    "convert_2d_to_3d",
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


def count_parameters(
    model: nn.Module, keys: Optional[Sequence[str]] = None, human_friendly: bool = False
) -> Dict[str, int]:
    """
    Count number of total and trainable parameters of a model
    :param model: A model
    :param keys: Optional list of top-level blocks
    :param human_friendly: If True, outputs human-friendly number of paramters: 13.3M, 124K
    :return: Tuple (total, trainable)
    """
    if keys is None:
        keys = [key for key, child in model.named_children()]
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    parameters = {"total": total, "trainable": trainable}

    for key in keys:
        try:
            if hasattr(model, key) and model.__getattr__(key) is not None:
                parameters[key] = int(sum(p.numel() for p in model.__getattr__(key).parameters()))
        except AttributeError:
            pass

    if human_friendly:
        for key in parameters.keys():
            parameters[key] = int_to_string_human_friendly(parameters[key])
    return parameters


def int_to_string_human_friendly(value: int) -> str:
    if value < 1000:
        return str(value)
    if value < 1000000:
        return f"{value / 1000.:.2f}K"
    if value < 10000000:
        return f"{value / 1000000.:.2f}M"
    if value < 100000000:
        return f"{value / 1000000.:.1f}M"
    if value < 1000000000:
        return f"{value / 1000000.:.1f}M"
    return f"{value / 1000000000.:.2f}B"


def to_numpy(x: Union[torch.Tensor, np.ndarray, Any, None]) -> Union[np.ndarray, None]:
    """
    Convert whatever to numpy array. None value returned as is.

    Args:
        :param x: List, tuple, PyTorch tensor or numpy array

    Returns:
        :return: Numpy array
    """
    if x is None:
        return None
    elif torch.is_tensor(x) and x.dtype == torch.bfloat16:
        return x.data.float().cpu().numpy()
    elif torch.is_tensor(x):
        return x.data.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (Iterable, int, float)):
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


def container_to_tensor(value: Union[np.ndarray, List, Tuple, Mapping, Any]):
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind not in {"O", "M", "U", "S"}:
        return torch.from_numpy(value)
    if isinstance(value, list):
        return [container_to_tensor(item) for item in value]
    if isinstance(value, tuple):
        return tuple(container_to_tensor(item) for item in value)
    if isinstance(value, Mapping):
        cls = type(value)
        return cls((k, container_to_tensor(v)) for k, v in value.items())

    raise ValueError(f"Unsupported container type {type(value)}")


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
    image = np.require(image, requirements="C")
    image = torch.from_numpy(image)
    return image


@pytorch_toolbelt_deprecated("This function is deprecated, please use image_to_tensor instead")
def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    return image_to_tensor(image)


@pytorch_toolbelt_deprecated("This function is deprecated, please use image_to_tensor instead")
def tensor_from_mask_image(mask: np.ndarray) -> torch.Tensor:
    return image_to_tensor(mask)


def rgb_image_from_tensor(
    image: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    min_pixel_value=0.0,
    max_pixel_value=255.0,
    dtype=np.uint8,
) -> np.ndarray:
    """
    Convert numpy image (RGB, BGR, Grayscale, SAR, Mask image, etc.) to tensor

    Args:
        image: A torch tensor of [C,H,W] shape
    """
    image = np.moveaxis(to_numpy(image), 0, -1)
    mean = to_numpy(mean)
    std = to_numpy(std)
    rgb_image = max_pixel_value * (image * std + mean)
    rgb_image = np.clip(rgb_image, a_min=min_pixel_value, a_max=max_pixel_value)
    return rgb_image.astype(dtype)


def mask_from_tensor(mask: torch.Tensor, squeeze_single_channel: bool = False, dtype=None) -> np.ndarray:
    mask_np = np.moveaxis(to_numpy(mask), 0, -1)
    if squeeze_single_channel and mask_np.shape[-1] == 1:
        mask_np = np.squeeze(mask_np, -1)

    if dtype is not None:
        mask_np = mask_np.astype(dtype)
    return mask_np


def maybe_cuda(x: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
    """
    Move input Tensor or Module to CUDA device if CUDA is available.
    :param x:
    :return:
    """
    if torch.cuda.is_available():
        return x.cuda()
    return x



@dataclasses.dataclass
class TransferWeightsOuptut:
    """
    Output of transfer_weights function. Holds information about how many layers were loaded, skipped, etc.
    Can be used to get detailed information about how many layers were loaded from checkpoint to model.
    """
    loaded_layers: List[str]
    skipped_layers: List[str]
    missing_layers_in_model: List[str]
    missing_layers_in_checkpoint: List[str]

    def __repr__(self):
        total_layers_in_checkpoint = len(self.loaded_layers) + len(self.missing_layers_in_model) + len(self.skipped_layers)
        total_layers_in_model = len(self.loaded_layers) + len(self.missing_layers_in_checkpoint) + len(self.skipped_layers)
        loaded_layers_percentage = 100.0 * len(self.loaded_layers) / total_layers_in_checkpoint
        skipped_layers_percentage = 100.0 * len(self.skipped_layers) / total_layers_in_checkpoint
        model_initialized_percentage = 100.0 * len(self.loaded_layers) / total_layers_in_model
        return f"TransferWeightsOuptut({model_initialized_percentage=:.2f}, {loaded_layers_percentage=:.2f}, {skipped_layers_percentage=:.2f})"

def transfer_weights(model: nn.Module, model_state_dict: collections.OrderedDict, incompatible_shape_action="skip") ->TransferWeightsOuptut:
    """
    Copy weights from state dict to model, skipping layers that are incompatible.
    This method is helpful if you are doing some model surgery and want to load
    part of the model weights into different model.
    :param model: Model to load weights into
    :param model_state_dict: Model state dict to load weights from
    :param incompatible_shape_action: What to do if shape of weight tensor is incompatible.
           Possible values are:
               - "skip" - Skip loading this tensor
               - "match_mean_std" - Initialize tensor with random values with same mean and std as source tensor
    :return: Instance of TransferWeightsOuptut
    """
    existing_model_state_dict = model.state_dict()

    loaded_layer_names = []
    skipped_layers_names = []
    layers_not_in_model = list(set(existing_model_state_dict.keys()) - set(model_state_dict.keys()))
    layers_not_in_checkpoint = list(set(model_state_dict.keys()) - set(existing_model_state_dict.keys()))

    for name, value in model_state_dict.items():
        if name not in existing_model_state_dict:
            logger.debug(
                f"transfer_weights skipped loading weights for key {name}, because it does not exist in model"
            )
            continue

        existing_value = existing_model_state_dict[name]
        if value.shape != existing_value.shape:
            if incompatible_shape_action == "skip":
                skipped_layers_names.append(name)
                logger.debug(
                    f"transfer_weights skipped loading weights for key {name}, because of checkpoint has shape {value.shape} and model has shape {existing_model_state_dict[name].shape}"
                )
                continue
            elif incompatible_shape_action == "match_mean_std":
                logger.debug(
                    f"transfer_weights found that {name} weights tensor have incompatible shape {value.shape} and model has shape {existing_value.shape}. "
                    f"Initializing with random values with same mean {existing_value.mean()} and std {existing_value.std()} from corresponding checkpoint weights tensor."
                )
                torch.nn.init.normal_(existing_value, mean=value.mean(), std=value.std())
                value = existing_value
            else:
                raise ValueError(f"Unsupported incompatible_shape_action={incompatible_shape_action}")

        try:
            model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
            loaded_layer_names.append(name)
        except Exception as e:
            logger.debug(f"transfer_weights skipped loading weights for key {name}, because of error: {e}")

    return TransferWeightsOuptut(
        loaded_layers=loaded_layer_names,
        skipped_layers=skipped_layers_names,
        missing_layers_in_model=layers_not_in_model,
        missing_layers_in_checkpoint=layers_not_in_checkpoint,
    )


def resize_like(x: Tensor, target: Tensor, mode: str = "bilinear", align_corners: Union[bool, None] = True) -> Tensor:
    """
    Resize input tensor to have the same spatial dimensions as target.

    Args:
        x: Input tensor of [B,C,H,W]
        target: [Bt,Ct,Ht,Wt]
        mode:
        align_corners:

    Returns:
        Resized tensor [B,C,Ht,Wt]
    """
    return torch.nn.functional.interpolate(x, target.size()[2:], mode=mode, align_corners=align_corners)


def move_to_device_non_blocking(x: Tensor, device: torch.device) -> Tensor:
    return move_to_device(x, device, non_blocking=True)


def move_to_device(
    x: Union[Tensor, List[Tensor], Tuple[Tensor, ...], Dict[Any, Tensor]], device: torch.device, non_blocking=False
) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=non_blocking)
    elif isinstance(x, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in x)
    elif isinstance(x, list):
        return [move_to_device(item, device, non_blocking) for item in x]
    elif isinstance(x, Mapping):
        cls = type(x)
        return cls((key, move_to_device(item, device, non_blocking)) for key, item in x.items())
    return x


resize_as = resize_like


def describe_outputs(outputs: Union[Tensor, Dict[str, Tensor], Iterable[Tensor]]) -> Union[List[Dict], Dict[str, Any]]:
    """
    Describe outputs and return shape, mean & std for each tensor in list or dict (Supports nested tensors)

    Args:
        outputs: Input (Usually model outputs)
    Returns:
        Same structure but each item represents tensor shape, mean & std
    """
    if torch.is_tensor(outputs):
        if torch.is_floating_point(outputs):
            desc = dict(
                size=tuple(outputs.size()), mean=outputs.mean().item(), std=outputs.std().item(), dtype=outputs.dtype
            )
        else:
            desc = dict(size=tuple(outputs.size()), num_unique=len(torch.unique(outputs)), dtype=outputs.dtype)
    elif isinstance(outputs, collections.abc.Mapping):
        desc = {}
        for key, value in outputs.items():
            desc[key] = describe_outputs(value)
    elif dataclasses.is_dataclass(outputs):
        desc = dataclasses.asdict(outputs)
        for key, value in desc.items():
            desc[key] = describe_outputs(value)
    elif isinstance(outputs, collections.abc.Iterable):
        desc = []
        for index, output in enumerate(outputs):
            desc.append(describe_outputs(output))
    else:
        warnings.warn(f"describe_outputs is not implemeted for type {type(outputs)}")
        return str(outputs)
    return desc


def get_collate_for_dataset(
    dataset: Union[Dataset, ConcatDataset], ensure_collate_fn_are_the_same: bool = True
) -> Callable:
    """
    Return collate_fn function for dataset. By default, default_collate returned.
    If the dataset has method get_collate_fn() we will use it's return value instead.
    If the dataset is ConcatDataset, we will check whether all get_collate_fn() returns
    the same function.

    Args:
        dataset: Input dataset

    Returns:
        Collate function to put into DataLoader
    """
    collate_fn = default_collate

    if hasattr(dataset, "get_collate_fn"):
        return dataset.get_collate_fn()
    elif isinstance(dataset, ConcatDataset):
        collate_fns = [get_collate_for_dataset(ds) for ds in dataset.datasets]
        collate_fn = collate_fns[0]

        if ensure_collate_fn_are_the_same:
            for other_collate_fn in collate_fns[1:]:
                if type(other_collate_fn) != type(collate_fn):
                    raise ValueError(
                        f"Detected ConcatDataset consist of datasets with different collate functions: {type(collate_fn)} and {type(other_collate_fn)}."
                    )

                if isinstance(collate_fn, functools.partial):
                    if not _partial_functions_equal(collate_fn, other_collate_fn):
                        raise ValueError(
                            f"Detected ConcatDataset consist of datasets with different collate functions: {collate_fn} and {type(other_collate_fn)}."
                        )
                elif collate_fn != other_collate_fn:
                    raise ValueError(
                        f"Detected ConcatDataset consist of datasets with different collate functions: {collate_fn} and {other_collate_fn}."
                    )

        collate_fn = collate_fns[0]

    return collate_fn


def _partial_functions_equal(func1, func2):
    if not (isinstance(func1, functools.partial) and isinstance(func2, functools.partial)):
        return False
    are_equal = all([getattr(func1, attr) == getattr(func2, attr) for attr in ["func", "args", "keywords"]])
    return are_equal


def get_non_wrapped_model(model: nn.Module) -> nn.Module:
    """
    Return real model from (maybe) wrapped in DP / DDP

    Args:
        model:

    Returns:

    """
    from torch.nn import DataParallel
    from torch.nn.parallel import DistributedDataParallel

    if not isinstance(model, nn.Module):
        raise RuntimeError("Input model must be a subclass of nn.Module.")

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module

    return model


def convert_2d_to_3d(model: nn.Module) -> nn.Module:
    """
    Recursively convert all 2d layers in `model` to their 3d versions.
    This method converts 2D CNN model to 3D version. Important note - models/layers with non-trivial forward() method
    probably not going to work (LayerNorm2d or GlobalResponseNormalization for instance)
    Replicates the existing Conv2d weights along the 3rd dimension (depth=1 by default) and scales them accordingly.
    
    :param model: Model to convert
    """
    for name, module in model.named_children():

        # If we find a Conv2d, replace it with a Conv3d.
        if isinstance(module, nn.Conv2d):
            # --------------------------------------------
            # 1) Check that the 2D kernel is square
            # --------------------------------------------
            if module.kernel_size[0] != module.kernel_size[1]:
                raise ValueError(
                    f"Non-square kernel detected: {module.kernel_size}. " "This example only handles square kernels (k, k)."
                )
            k = module.kernel_size[0]

            # --------------------------------------------
            # 2) Build a new Conv3d with kernel_size = (k, k, k)
            #    using the same hyperparameters as best we can
            # --------------------------------------------
            new_conv = nn.Conv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=(k, k, k),
                # For stride, padding, and dilation, we replicate
                # the 2D values in each dimension:
                stride=(module.stride[0], module.stride[0], module.stride[1]),
                padding=(module.padding[0], module.padding[0], module.padding[1]),
                dilation=(module.dilation[0], module.dilation[0], module.dilation[1]),
                groups=module.groups,
                bias=(module.bias is not None),
            )

            # --------------------------------------------
            # 3) Copy and replicate the 2D weights -> 3D
            # old_weight shape: (out_c, in_c, k, k)
            # new_weight shape: (out_c, in_c, k, k, k)
            # --------------------------------------------
            with torch.no_grad():
                old_weight = module.weight  # shape: (out_c, in_c, k, k)
                # Expand along a new depth dimension
                old_weight_3d = old_weight.unsqueeze(2)  # (out_c, in_c, 1, k, k)
                old_weight_3d = old_weight_3d.repeat(1, 1, k, 1, 1).div(k)  # (out_c, in_c, k, k, k)
                new_conv.weight.copy_(old_weight_3d)

                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)

            # Replace the old Conv2d with our new Conv3d
            setattr(model, name, new_conv)

        # If we find a BatchNorm2d, replace it with a BatchNorm3d.
        elif isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm3d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )

            # Copy running statistics and affine parameters
            with torch.no_grad():
                if module.affine:
                    new_bn.weight.copy_(module.weight)
                    new_bn.bias.copy_(module.bias)
                new_bn.running_mean.copy_(module.running_mean)
                new_bn.running_var.copy_(module.running_var)

            # Replace the BatchNorm2d with BatchNorm3d
            setattr(model, name, new_bn)
        elif isinstance(module, nn.ReLU) and replace_relu_with_silu:
            # Replace with SILU
            setattr(model, name, nn.SiLU(inplace=True))
        elif isinstance(module, nn.Dropout2d):
            # Replace with Dropout3d
            setattr(model, name, nn.Dropout3d(p=module.p, inplace=module.inplace))
        else:
            # Recursively convert children
            convert_2d_to_3d(module)

    return model
