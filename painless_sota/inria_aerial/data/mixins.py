import abc
from typing import Dict

import cv2
import numpy as np
from skimage.measure import label
from torch import Tensor

from painless_sota.inria_aerial.data.functional import remove_small_instances, compute_weighting_mask
from pytorch_toolbelt.datasets import (
    TARGET_MASK_KEY,
    TARGET_MASK_WEIGHT_KEY,
    TARGET_MASK_KEY_STRIDE_64,
    TARGET_MASK_KEY_STRIDE_8,
    TARGET_MASK_KEY_STRIDE_2,
    TARGET_MASK_KEY_STRIDE_4,
    TARGET_MASK_KEY_STRIDE_16,
    TARGET_MASK_KEY_STRIDE_32,
)
from pytorch_toolbelt.utils import image_to_tensor

__all__ = ["SegmentationTargetMixin", "TransformationsMixin", "TargetMixin"]


class TransformationsMixin:
    def apply_transformation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        min_instance_size: int,
        min_instance_visibility: float,
    ):
        """
        Apply image augmentations and postprocessing to remove small instances from the target mask

        :param image:
        :param mask:
        :param min_instance_size:
        :param min_instance_visibility:
        :return:
        """
        # Count instances before augmentations
        instances_pre = dict(zip(*np.unique(label(mask), return_counts=True)))

        data = self.transform(image=image, mask=mask)

        image = data["image"]
        mask = data["mask"]

        # Count instances after augmentation
        instances_post = dict(zip(*np.unique(label(mask), return_counts=True)))

        # Keep only those who retained more than certain percentage of their area
        if min_instance_visibility > 0:
            dropout_mask = np.zeros_like(mask, dtype=np.bool)
            for instance_label, instance_area_post in instances_post.items():
                if instance_label == 0:
                    continue  # Ignore background

                instance_area_pre = instances_pre.get(instance_label, 0)
                if instance_area_pre:
                    ratio = float(instance_area_post) / float(instance_area_pre)
                    if ratio < min_instance_visibility:
                        dropout_mask |= mask == instance_label

            mask[dropout_mask] = 0

        if min_instance_size > 0:
            mask = remove_small_instances(mask, min_size=min_instance_size, inplace=True)

        return image, mask


class TargetMixin:
    @abc.abstractmethod
    def compute_targets(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Tensor]:
        raise NotImplementedError()


class SegmentationTargetMixin(TargetMixin):
    def __init__(self, need_supervision_masks: bool = False, need_weighting_mask: bool = False):
        self.need_supervision_masks = need_supervision_masks
        self.need_weighting_mask = need_weighting_mask

    def compute_targets(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Tensor]:
        _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)

        targets = {
            TARGET_MASK_KEY: image_to_tensor(mask, dummy_channels_dim=True).float(),  # [1,H,W]
        }

        if self.need_supervision_masks:
            mask_i = mask
            for key in [
                TARGET_MASK_KEY_STRIDE_2,
                TARGET_MASK_KEY_STRIDE_4,
                TARGET_MASK_KEY_STRIDE_8,
                TARGET_MASK_KEY_STRIDE_16,
                TARGET_MASK_KEY_STRIDE_32,
                TARGET_MASK_KEY_STRIDE_64,
            ]:
                mask_i = cv2.pyrDown(mask_i)
                _, mask_binary = cv2.threshold(mask_i, 0, 1, cv2.THRESH_BINARY)
                targets[key] = image_to_tensor(mask_binary, dummy_channels_dim=True).float()  # [1,H,W]

        if self.need_weighting_mask:
            weight_mask = compute_weighting_mask(mask)
            targets[TARGET_MASK_WEIGHT_KEY] = image_to_tensor(weight_mask, dummy_channels_dim=True).float()  # [1,H,W]

        return targets
