from typing import Optional, List

import albumentations as A
import torch
from torch.utils.data import Dataset

from .common import read_image_rgb, INPUT_IMAGE_KEY, INPUT_IMAGE_ID_KEY, INPUT_INDEX_KEY, TARGET_CLASS_KEY
from ..utils import fs, image_to_tensor

__all__ = ["ClassificationDataset", "label_to_tensor"]


def label_to_tensor(x):
    return torch.tensor(x).long()


class ClassificationDataset(Dataset):
    """
    Dataset for image classification tasks
    """

    def __init__(
        self,
        image_filenames: List[str],
        labels: Optional[List[str]],
        transform: A.Compose,
        read_image_fn=read_image_rgb,
        make_target_fn=label_to_tensor,
    ):
        if labels is not None and len(image_filenames) != len(labels):
            raise ValueError("Number of images does not corresponds to number of targets")

        self.image_ids = [fs.id_from_fname(fname) for fname in image_filenames]
        self.labels = labels
        self.images = image_filenames
        self.read_image = read_image_fn
        self.transform = transform
        self.make_target = make_target_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.read_image(self.images[index])
        data = self.transform(image=image)

        image = data["image"]

        sample = {
            INPUT_INDEX_KEY: index,
            INPUT_IMAGE_ID_KEY: self.image_ids[index],
            INPUT_IMAGE_KEY: image_to_tensor(image),
        }

        if self.labels is not None:
            sample[TARGET_CLASS_KEY] = self.make_target(self.labels[index])
        return sample
