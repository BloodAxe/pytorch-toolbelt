import random
import typing
from typing import Any, Optional

from torch.utils.data import Dataset
import numpy as np

__all__ = ["RandomSubsetDataset", "RandomSubsetWithMaskDataset"]

from torch.utils.data.dataloader import default_collate


class RandomSubsetDataset(Dataset):
    """
    Wrapper to get desired number of samples from underlying dataset
    """

    def __init__(self, dataset, num_samples: int, weights: Optional[np.ndarray] = None):
        if weights is not None:
            if len(dataset) != len(weights):
                raise ValueError(
                    "Length of weights must be equal to length of dataset. Got {} and {}".format(
                        len(weights), len(dataset)
                    )
                )
        self.dataset = dataset
        self.num_samples = num_samples
        self.weights = np.cumsum(weights) if weights is not None else None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _) -> Any:
        if self.weights is not None:
            population = range(len(self.dataset))
            index = random.choices(population, cum_weights=self.weights, k=1)[0]
        else:
            index = random.randrange(len(self.dataset))
        return self.dataset[index]

    def get_collate_fn(self):
        get_collate_fn = getattr(self.dataset, "get_collate_fn", None)
        if callable(get_collate_fn):
            return get_collate_fn()
        return default_collate


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

    def get_collate_fn(self) -> typing.Callable:
        get_collate_fn = getattr(self.dataset, "get_collate_fn", None)
        if callable(get_collate_fn):
            return get_collate_fn()
        return default_collate
