from functools import partial
from typing import Optional, List, Callable

import albumentations as A
import cv2
import numpy as np
from skimage.measure import block_reduce
from torch.utils.data import Dataset

from .common import (
    read_image_rgb,
    INPUT_IMAGE_KEY,
    INPUT_IMAGE_ID_KEY,
    INPUT_INDEX_KEY,
    TARGET_MASK_WEIGHT_KEY,
    TARGET_MASK_KEY,
    name_for_stride,
    UNLABELED_SAMPLE,
)
from ..utils import fs, image_to_tensor

__all__ = ["mask_to_bce_target", "mask_to_ce_target", "SegmentationDataset", "compute_weight_mask"]


def mask_to_bce_target(mask):
    return image_to_tensor(mask, dummy_channels_dim=True).float()


def mask_to_ce_target(mask):
    return image_to_tensor(mask, dummy_channels_dim=False).long()


def compute_weight_mask(mask: np.ndarray, edge_weight=4) -> np.ndarray:
    from skimage.morphology import binary_dilation, binary_erosion

    binary_mask = mask > 0
    weight_mask = np.ones(mask.shape[:2]).astype(np.float32)

    if binary_mask.any():
        dilated = binary_dilation(binary_mask, structure=np.ones((5, 5), dtype=np.bool))
        eroded = binary_erosion(binary_mask, structure=np.ones((5, 5), dtype=np.bool))

        a = dilated & ~binary_mask
        b = binary_mask & ~eroded

        weight_mask = (a | b).astype(np.float32) * edge_weight + 1
        weight_mask = cv2.GaussianBlur(weight_mask, ksize=(5, 5), sigmaX=5)
    return weight_mask


def _block_reduce_dominant_label(x: np.ndarray, axis):
    try:
        # minlength is +1 to num classes because we must account for IGNORE_LABEL
        minlength = np.max(x) + 1
        bincount_fn = partial(np.bincount, minlength=minlength)
        counts = np.apply_along_axis(bincount_fn, -1, x.reshape((x.shape[0], x.shape[1], -1)))
        reduced = np.argmax(counts, axis=-1)
        return reduced
    except Exception as e:
        print(e)
        print("shape", x.shape, "axis", axis)


def read_binary_mask(mask_fname: str) -> np.ndarray:
    mask = cv2.imread(mask_fname, cv2.IMREAD_COLOR)
    return cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY, dst=mask)


class SegmentationDataset(Dataset):
    """
    Dataset class suitable for segmentation tasks
    """

    def __init__(
        self,
        image_filenames: List[str],
        mask_filenames: Optional[List[str]],
        transform: A.Compose,
        read_image_fn: Callable = read_image_rgb,
        read_mask_fn: Callable = cv2.imread,
        need_weight_mask=False,
        need_supervision_masks=False,
        make_mask_target_fn: Callable = mask_to_ce_target,
    ):
        if mask_filenames is not None and len(image_filenames) != len(mask_filenames):
            raise ValueError("Number of images does not corresponds to number of targets")

        self.image_ids = [fs.id_from_fname(fname) for fname in image_filenames]
        self.need_weight_mask = need_weight_mask
        self.need_supervision_masks = need_supervision_masks

        self.images = image_filenames
        self.masks = mask_filenames
        self.read_image = read_image_fn
        self.read_mask = read_mask_fn

        self.transform = transform
        self.make_target = make_mask_target_fn

    def __len__(self):
        return len(self.images)

    def set_target(self, index: int, value: np.ndarray):
        mask_fname = self.masks[index]

        value = (value * 255).astype(np.uint8)
        cv2.imwrite(mask_fname, value)

    def __getitem__(self, index):
        image = self.read_image(self.images[index])

        if self.masks is not None:
            mask = self.read_mask(self.masks[index])
        else:
            mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.uint8) * UNLABELED_SAMPLE

        data = self.transform(image=image, mask=mask)

        image = data["image"]
        mask = data["mask"]

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: self.image_ids[index],
            INPUT_IMAGE_KEY: image_to_tensor(image),
            TARGET_MASK_KEY: self.make_target(mask),
        }

        if self.need_weight_mask:
            sample[TARGET_MASK_WEIGHT_KEY] = image_to_tensor(compute_weight_mask(mask)).float()

        if self.need_supervision_masks:
            for i in range(1, 5):
                stride = 2 ** i
                mask = block_reduce(mask, (2, 2), partial(_block_reduce_dominant_label))
                sample[name_for_stride(TARGET_MASK_KEY, stride)] = self.make_target(mask)

        return sample
