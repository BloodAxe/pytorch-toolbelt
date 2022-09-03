import math
import typing
from functools import partial
from typing import Dict, List, Any, Callable, Tuple, Union

import albumentations as A
import pandas as pd
from omegaconf import DictConfig
from painless_sota.inria_aerial.data.functional import (
    as_tuple_of_two,
    inria_collate_fn,
    INPUT_SCENE_PATH_KEY,
    INPUT_SCENE_CROP_KEY,
)
from painless_sota.inria_aerial.data.image_io import ImageAnnotation
from painless_sota.inria_aerial.data.mixins import TransformationsMixin, SegmentationTargetMixin, TargetMixin
from pytorch_toolbelt.datasets import (
    INPUT_IMAGE_KEY,
    INPUT_INDEX_KEY,
    INPUT_IMAGE_ID_KEY,
)
from pytorch_toolbelt.utils import image_to_tensor
from torch.utils.data import Dataset

__all__ = ["FixedCropFromImageDataset"]


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
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        image_size: Union[int, Tuple[int, int]],
        channels_last: bool,
        transforms: DictConfig,
        need_supervision_masks: bool = False,
        need_weighting_mask: bool = False,
        min_instance_size: int = 0,
    ):
        image_size: Tuple[int, int] = typing.cast(Tuple[int, int], as_tuple_of_two(image_size))

        tiled_annotations = []
        crop_coodinates = []

        transform = A.Compose(
            [
                *(transforms.augmentations if transforms.augmentations is not None else []),
                *transforms.size,
                *transforms.normalization,
            ]
        )

        for index, row in df.iterrows():
            image_shape = row["rows"], row["cols"]
            crops = list(cls.iterate_crops(image_shape, image_size))
            ann = ImageAnnotation(row)

            for crop_coords in crops:
                tiled_annotations.append(ann)
                crop_coodinates.append(crop_coords)

        return cls(
            annotations=tiled_annotations,
            crop_coords=crop_coodinates,
            target_mixins=[
                SegmentationTargetMixin(
                    need_supervision_masks=need_supervision_masks, need_weighting_mask=need_weighting_mask
                )
            ],
            transform=transform,
            channels_last=channels_last,
            min_instance_size=min_instance_size,
        )

    def __init__(
        self,
        annotations: List[ImageAnnotation],
        crop_coords: List[Tuple],
        target_mixins: List[TargetMixin],
        transform: A.Compose,
        min_instance_size: int,
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

    def __len__(self):
        return len(self.annotations)

    def sample_crop_area(self, index: int):
        crop_coords = self.crop_coords[index]
        return crop_coords

    def __getitem__(self, index: int) -> Dict[str, Any]:
        crop_coords = self.sample_crop_area(index)
        image, mask = self.annotations[index].load_image(crop_coords)

        image, mask = self.apply_transformation(
            image,
            mask,
            min_instance_visibility=0.1,
            min_instance_size=self.min_instance_size,
        )

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_KEY: image_to_tensor(image),
            INPUT_IMAGE_ID_KEY: self.annotations[index].image_id,
            INPUT_SCENE_PATH_KEY: self.annotations[index].image_path,
            INPUT_SCENE_CROP_KEY: crop_coords,
        }

        for mixin in self.target_mixins:
            targets = mixin.compute_targets(image, mask)
            sample.update(targets)

        return sample

    def get_collate_fn(self) -> Callable:
        return partial(inria_collate_fn, channels_last=self.channels_last)
