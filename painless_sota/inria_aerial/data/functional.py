import numbers
from typing import Tuple, Union, Iterable

import cv2
import numpy as np
import torch
from skimage.measure import label
from skimage.segmentation import find_boundaries
from torch.utils.data.dataloader import default_collate

from pytorch_toolbelt.datasets import (
    INPUT_IMAGE_KEY, INPUT_IMAGE_ID_KEY,
)

__all__ = [
    "compute_weighting_mask",
    "remove_small_instances",
    "inria_collate_fn",
    "as_tuple_of_two",
    "INPUT_SCENE_CROP_KEY",
    "INPUT_SCENE_PATH_KEY",
]

INPUT_SCENE_CROP_KEY = "INPUT_SCENE_CROP_KEY"
INPUT_SCENE_PATH_KEY = "INPUT_SCENE_PATH_KEY"


def compute_weighting_mask(binary_mask: np.ndarray) -> np.ndarray:
    binary_mask = binary_mask > 0
    edge_mask = find_boundaries(binary_mask, mode="thick")
    edge_mask_f32 = edge_mask.astype(np.float32)
    edge_mask_f32 = cv2.blur(edge_mask_f32, ksize=(5, 5))
    edge_mask_f32 = cv2.blur(edge_mask_f32, ksize=(5, 5))
    edge_mask_f32[edge_mask] = 1.0

    weight_mask = binary_mask.astype(np.float32) + edge_mask_f32
    return weight_mask


def remove_small_instances(semantic_mask: np.ndarray, min_size: int, inplace=False) -> np.ndarray:
    """
    Remove small instances from semantic mask.
    This function replaces remove_small_objects from skimage.morphology since the latter
    works incorrect when multiple instances of same semantic label is present (they contribute to label area as a single object).

    This implementation changes this and also supports inplace modification.

    :param semantic_mask: Input mask [H,W] of np.uint8 dtype
    :param min_size: Min instance size
    :param inplace: Whether to operate inplace
    :return:
    """
    labels = label(semantic_mask)
    component_sizes = np.bincount(labels.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labels]

    if inplace:
        out = semantic_mask
    else:
        out = semantic_mask.copy()

    out[too_small_mask] = 0
    return out


def inria_collate_fn(batch, channels_last=False):
    skip_keys = [
        INPUT_IMAGE_ID_KEY,
        INPUT_SCENE_PATH_KEY,
        INPUT_SCENE_CROP_KEY,
    ]
    excluded_items = [dict((k, v) for k, v in b.items() if k in skip_keys) for b in batch]
    included_items = [dict((k, v) for k, v in b.items() if k not in skip_keys) for b in batch]

    batch: dict = default_collate(included_items)
    for k in skip_keys:
        out = [item[k] for item in excluded_items if k in item]
        if len(out):
            batch[k] = out

    if channels_last:
        batch[INPUT_IMAGE_KEY] = batch[INPUT_IMAGE_KEY].to(memory_format=torch.channels_last)
    return batch


def as_tuple_of_two(
    value: Union[numbers.Number, Tuple[numbers.Number, numbers.Number]]
) -> Tuple[numbers.Number, numbers.Number]:
    """
    Takes single number of tuple of two numbers and always returns a tuple of two numbers
    Args:
        value:

    Returns:
        512     - (512,512)
        256,257 - (256, 257)
    """
    if isinstance(value, Iterable):
        a, b = value
        return a, b
    if isinstance(value, numbers.Number):
        return value, value
    raise RuntimeError(f"Unsupported input value {value}")
