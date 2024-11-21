from functools import partial

import cv2
import numpy as np

from ..utils import image_to_tensor

__all__ = ["mask_to_bce_target", "mask_to_ce_target", "read_binary_mask", "compute_weight_mask"]


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
