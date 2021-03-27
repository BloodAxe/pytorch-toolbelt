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
)
from ..utils import fs, image_to_tensor

__all__ = ["mask_to_bce_target", "mask_to_ce_target", "read_binary_mask", "SegmentationDataset", "compute_weight_mask"]


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
    """
    Read image as binary mask, all non-zero values are treated as positive labels and converted to 1
    Args:
        mask_fname: Image with mask

    Returns:
        Numpy array with {0,1} values
    """

    mask = cv2.imread(mask_fname, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot find {mask_fname}")

    cv2.threshold(mask, thresh=0, maxval=1, type=cv2.THRESH_BINARY, dst=mask)
    return mask


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
        image_ids: Optional[List[str]] = None,
    ):
        if mask_filenames is not None and len(image_filenames) != len(mask_filenames):
            raise ValueError("Number of images does not corresponds to number of targets")

        if image_ids is None:
            self.image_ids = [fs.id_from_fname(fname) for fname in image_filenames]
        else:
            self.image_ids = image_ids

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

    def __getitem__(self, index):
        image = self.read_image(self.images[index])
        data = {"image": image}
        if self.masks is not None:
            data["mask"] = self.read_mask(self.masks[index])

        data = self.transform(**data)

        image = data["image"]
        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: self.image_ids[index],
            INPUT_IMAGE_KEY: image_to_tensor(image),
        }

        if self.masks is not None:
            mask = data["mask"]
            sample[TARGET_MASK_KEY] = self.make_target(mask)
            if self.need_weight_mask:
                sample[TARGET_MASK_WEIGHT_KEY] = image_to_tensor(compute_weight_mask(mask)).float()

            if self.need_supervision_masks:
                for i in range(1, 6):
                    stride = 2 ** i
                    mask = block_reduce(mask, (2, 2), partial(_block_reduce_dominant_label))
                    sample[name_for_stride(TARGET_MASK_KEY, stride)] = self.make_target(mask)

        return sample
