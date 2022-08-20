import math
from functools import partial
from typing import Dict, Mapping, List, Any, Callable, Tuple
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from pytorch_toolbelt.datasets import (
    TARGET_MASK_KEY,
    TARGET_MASK_WEIGHT_KEY,
    TARGET_MASK_2_KEY,
    TARGET_MASK_16_KEY,
    TARGET_MASK_32_KEY,
    TARGET_MASK_4_KEY,
    TARGET_MASK_8_KEY,
    TARGET_MASK_64_KEY,
    INPUT_IMAGE_KEY,
)
from pytorch_toolbelt.utils import image_to_tensor
from skimage.segmentation import find_boundaries
from torch import Tensor

__all__ = ["SegmentationTargetMixin", "compute_weighting_mask"]


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
        INPUT_SCENE_ID_KEY,
        INPUT_SCENE_PATH_KEY,
        INPUT_SCENE_CROP_KEY,
        INPUT_INSTANCE_TO_CLASS_LABEL_KEY,
        INPUT_INSTANCE_MASK_KEY,
        TARGET_GROUNDTUTH_POLYGONS,
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


class TransformationsMixin:
    def apply_transformation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        min_instance_size: int,
        min_instance_visibility: float,
    ):
        """
        :param image:
        :param mask:
        :param min_instance_size:
        :param min_instance_visibility:
        :return:
        """
        # Count instances before
        instances_pre = dict(zip(*np.unique(mask, return_counts=True)))

        data = self.transform(image=image, mask=mask)

        image_post = data["image"]
        segmentation_mask_post = data["mask"]

        # Count instances after augmentation
        instance_mask_post = label(segmentation_mask_post)
        instances_post = dict(zip(*np.unique(instance_mask_post, return_counts=True)))

        # Keep only those who retained > 50% of their area
        if min_instance_visibility > 0:
            dropout_mask = np.zeros_like(segmentation_mask_post, dtype=np.bool)
            for instance_label, instance_area_post in instances_post.items():
                if instance_label == 0:
                    continue  # Ignore background

                instance_area_pre = instances_pre.get(instance_label, 0)
                if instance_area_pre:
                    ratio = float(instance_area_post) / float(instance_area_pre)
                    if ratio < min_instance_visibility:
                        dropout_mask |= segmentation_mask_post == instance_label

            segmentation_mask_post[dropout_mask] = 0

        if min_instance_size > 0:
            segmentation_mask_post = remove_small_instances(
                segmentation_mask_post, min_size=min_instance_size, inplace=True
            )

        return image_post, segmentation_mask_post


class SegmentationTargetMixin:
    def __init__(self, need_supervision_masks: bool = False, need_weighting_mask: bool = False):
        self.need_supervision_masks = need_supervision_masks
        self.need_weighting_mask = need_weighting_mask

    def compute_targets(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Tensor]:
        targets = {
            TARGET_MASK_KEY: image_to_tensor(
                cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY), dummy_channels_dim=True
            ).float(),  # [H,W,1]
        }

        if self.need_supervision_masks:
            mask_i = mask
            for key in [
                TARGET_MASK_2_KEY,
                TARGET_MASK_4_KEY,
                TARGET_MASK_8_KEY,
                TARGET_MASK_16_KEY,
                TARGET_MASK_32_KEY,
                TARGET_MASK_64_KEY,
            ]:
                mask_i = cv2.pyrDown(mask_i)
                targets[key] = image_to_tensor(
                    cv2.threshold(mask_i, 0, 1, cv2.THRESH_BINARY), dummy_channels_dim=True
                ).float()  # [H,W,1]

        if self.need_weighting_mask:
            weight_mask = compute_weighting_mask(mask)
            targets[TARGET_MASK_WEIGHT_KEY] = image_to_tensor(weight_mask, dummy_channels_dim=True).float()  # [H,W,1]

        return targets


class FixedCropFromImageDataset(Dataset, TransformationsMixin):
    @classmethod
    def iterate_crops(cls, image_shape: Tuple[int, int], tile_size: Tuple[int, int]):
        tile_rows, tile_cols = tile_size
        step_rows, step_cols = tile_size

        rows, cols = image_shape[:2]
        num_rows = int(math.ceil(rows / float(step_rows)))
        num_cols = int(math.ceil(cols / float(step_cols)))

        for row_index in range(num_rows):
            row_start = row_index * step_rows
            row_end = min(image_shape[0], row_start + tile_rows)
            for col_index in range(num_cols):
                col_start = col_index * step_cols
                col_end = min(image_shape[1], col_start + tile_cols)
                yield (row_start, row_end), (col_start, col_end)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, crop_size):
        tiled_annotations = []
        crop_coodinates = []

        for index, row in df.iterrows():
            image_shape = row["rows"], row["cols"]
            crops = list(cls.iterate_crops(image_shape, crop_size))
            for crop_coords in crops:
                tiled_annotations.append(row)
                crop_coodinates.append(crop_coords)

            transform = A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=crop_size[0], min_width=crop_size[1], border_mode=cv2.BORDER_CONSTANT, value=0
                    ),
                    A.Normalize(),
                ]
            )

            mixins = [DecodeGroundtruthMixin(self.segmentation_class_mapping)]
            if "detection" in tasks:
                mixins.append(DetectionTargetsMixin(box_coder, self.detection_class_mapping))
            if "segmentation" in tasks:
                mixins.append(SegmentationMixin(self.segmentation_class_mapping))

            return FixedCropFromImageDataset(
                annotations=tiled_annotations,
                crop_coords=crop_coodinates,
                target_mixins=mixins,
                transform=transform,
                channels_last=channels_last,
                min_instance_size=min_instance_size,
                read_image_fn=self.read_image_fn,
            )

    def __init__(
        self,
        annotations: List[ImageAnnotation],
        crop_coords: List[Tuple],
        target_mixins: List[MixinProtocol],
        transform: A.Compose,
        min_instance_size: int,
        read_image_fn: Callable,
        channels_last: bool = False,
    ):
        if len(annotations) != len(crop_coords):
            raise RuntimeError("Number of annotations does not equal to number of crop_coords")
        self.annotations = annotations
        self.crop_coords = crop_coords
        self.target_mixins = target_mixins
        self.transform = transform
        self.channels_last = channels_last
        self.min_instance_size = min_instance_size
        self.read_image_fn = read_image_fn

    def __len__(self):
        return len(self.annotations)

    def sample_crop_area(self, index: int):
        crop_coords = self.crop_coords[index]
        return crop_coords

    def __getitem__(self, index: int) -> Dict[str, Any]:
        crop_coords = self.sample_crop_area(index)
        image, instance_mask = self.annotations[index].load_image(crop_coords, read_image_fn=self.read_image_fn)

        image, instance_mask, instance_to_class_label = self.apply_transformation(
            image,
            instance_mask,
            min_instance_visibility=0.5,
            min_instance_size=self.min_instance_size,
        )

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_KEY: image_to_tensor(image),
            INPUT_SCENE_ID_KEY: self.annotations[index].image_id,
            INPUT_SCENE_PATH_KEY: self.annotations[index].image_path,
            INPUT_SCENE_CROP_KEY: crop_coords,
            INPUT_INSTANCE_MASK_KEY: instance_mask,
            INPUT_INSTANCE_TO_CLASS_LABEL_KEY: instance_to_class_label,
        }

        for mixin in self.target_mixins:
            targets = mixin.compute_targets(image, instance_mask, instance_to_class_label)
            sample.update(targets)

        return sample

    def get_collate_fn(self) -> Callable:
        return partial(inria_collate_fn, channels_last=self.channels_last)
