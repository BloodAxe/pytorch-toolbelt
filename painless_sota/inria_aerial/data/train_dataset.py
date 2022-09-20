import random
import typing
from functools import partial
from typing import Dict, List, Any, Callable, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from painless_sota.inria_aerial.data.functional import (
    as_tuple_of_two,
    inria_collate_fn,
    INPUT_SCENE_CROP_KEY,
    INPUT_SCENE_PATH_KEY,
)
from painless_sota.inria_aerial.data.image_io import ImageAnnotation
from painless_sota.inria_aerial.data.mixins import TransformationsMixin, SegmentationTargetMixin, TargetMixin
from pytorch_toolbelt.datasets import (
    INPUT_IMAGE_KEY,
    INPUT_INDEX_KEY,
    INPUT_IMAGE_ID_KEY,
)
from pytorch_toolbelt.utils import image_to_tensor
from torch.utils.data import Dataset, WeightedRandomSampler

__all__ = ["RandomCropFromImageDataset"]


class RandomCropFromImageDataset(Dataset, TransformationsMixin):
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        image_size: Union[int, Tuple[int, int]],
        channels_last: bool,
        transforms: DictConfig,
        num_samples: int,
        need_supervision_masks: bool = False,
        need_weighting_mask: bool = False,
        min_instance_size: int = 0,
    ) -> Tuple[Dataset, WeightedRandomSampler]:
        tiled_annotations = [ImageAnnotation(row) for index, row in df.iterrows()]
        image_size: Tuple[int, int] = typing.cast(Tuple[int, int], as_tuple_of_two(image_size))

        transform = A.Compose(
            [
                *transforms.augmentations,
                *transforms.size,
                *transforms.normalization,
            ]
        )

        dataset = cls(
            annotations=tiled_annotations,
            target_mixins=[
                SegmentationTargetMixin(
                    need_supervision_masks=need_supervision_masks, need_weighting_mask=need_weighting_mask
                )
            ],
            image_size=image_size,
            transform=transform,
            channels_last=channels_last,
            min_instance_size=min_instance_size,
        )
        sampler = WeightedRandomSampler(weights=np.ones(len(dataset)), num_samples=num_samples)
        return dataset, sampler

    def __init__(
        self,
        annotations: List[ImageAnnotation],
        target_mixins: List[TargetMixin],
        transform: A.Compose,
        image_size: Tuple[int, int],
        min_instance_size: int,
        channels_last: bool = False,
    ):
        self.annotations = annotations
        self.target_mixins = target_mixins
        self.transform = transform
        self.channels_last = channels_last
        self.min_instance_size = min_instance_size
        self.image_size = image_size
        self.image_size_with_extra = int(image_size[0] * 1.5), int(image_size[1] * 1.5)

    def __len__(self):
        return len(self.annotations)

    def sample_crop_area(self, index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        scene_rows, scene_cols = self.annotations[index].shape

        cx = random.randint(self.image_size[0] // 2, scene_cols - self.image_size[0] // 2)
        cy = random.randint(self.image_size[1] // 2, scene_rows - self.image_size[1] // 2)

        start_row = max(0, cy - self.image_size_with_extra[1] // 2)
        start_col = max(0, cx - self.image_size_with_extra[0] // 2)

        end_row = min(scene_rows, start_row + self.image_size_with_extra[1])
        end_col = min(scene_cols, start_col + self.image_size_with_extra[0])

        crop_coords = (start_row, end_row), (start_col, end_col)
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
