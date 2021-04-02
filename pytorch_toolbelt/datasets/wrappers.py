import random
from typing import Any

from torch.utils.data import Dataset
import numpy as np

__all__ = ["RandomSubsetDataset", "RandomSubsetWithMaskDataset"]


class RandomSubsetDataset(Dataset):
    """
    Wrapper to get desired number of samples from underlying dataset
    """

    def __init__(self, dataset, num_samples: int):
        self.dataset = dataset
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _) -> Any:
        index = random.randrange(len(self.dataset))
        return self.dataset[index]


class RandomSubsetWithMaskDataset(Dataset):
    """
    Wrapper to get desired number of samples from underlying dataset only considering
    samples P for which mask[P] equals True
    """

    def __init__(self, dataset: Dataset, mask: np.ndarray, num_samples: int):
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool
            or len(mask.shape) != 1
            or len(mask) != len(dataset)
        ):
            raise ValueError("Mask must be boolean 1-D numpy array")

        if not mask.any():
            raise ValueError("Mask must have at least one positive value")

        self.dataset = dataset
        self.mask = mask
        self.num_samples = num_samples
        self.indexes = np.flatnonzero(self.mask)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _) -> Any:
        index = random.choice(self.indexes)
        return self.dataset[index]
