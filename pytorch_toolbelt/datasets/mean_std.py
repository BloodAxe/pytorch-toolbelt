import numpy as np
from typing import Optional, Tuple

__all__ = ["DatasetMeanStdCalculator"]


class DatasetMeanStdCalculator:
    __slots__ = ["global_mean", "global_var", "n_items", "num_channels", "global_max", "global_min", "dtype"]

    """
    Class to calculate running mean and std of the dataset. It helps when whole dataset does not fit entirely in RAM.
    """

    def __init__(self, num_channels: int = 3, dtype=np.float64):
        """
        Create a new instance of DatasetMeanStdCalculator

        Args:
            num_channels: Number of channels in the image. Default value is 3
        """
        super().__init__()
        self.num_channels = num_channels
        self.global_mean = None
        self.global_var = None
        self.global_max = None
        self.global_min = None
        self.n_items = 0
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.global_mean = np.zeros(self.num_channels, dtype=self.dtype)
        self.global_var = np.zeros(self.num_channels, dtype=self.dtype)
        self.global_max = np.ones_like(self.global_mean) * float("-inf")
        self.global_min = np.ones_like(self.global_mean) * float("+inf")
        self.n_items = 0

    def accumulate(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """
        Compute mean and std of a single image and integrates it into global statistics
        Args:
            image: Input image (Must be HWC, with number of channels C equal to self.num_channels)
            mask: Optional mask to include only certain parts of image from statistics computation.
            Only non-zero elements will be included,
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if self.num_channels != image.shape[2]:
            raise RuntimeError(f"Number of channels in image must be {self.num_channels}, got {image.shape[2]}.")
        image = image.reshape((-1, self.num_channels))

        if mask is not None:
            mask = mask.reshape((mask.shape[0] * mask.shape[1]))
            image = image[mask, :]

            # In case the whole image is masked out, we exclude it entirely
            if len(image) == 0:
                return

        mean = np.mean(image, axis=0)
        std = np.std(image, axis=0)

        self.global_mean += np.squeeze(mean)
        self.global_var += np.squeeze(std) ** 2
        self.global_max = np.maximum(self.global_max, np.max(image, axis=0))
        self.global_min = np.minimum(self.global_min, np.min(image, axis=0))
        self.n_items += 1

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dataset-level mean & std

        Returns:
            Tuple of global [mean, std] per channel
        """
        return self.global_mean / self.n_items, np.sqrt(self.global_var / self.n_items)
