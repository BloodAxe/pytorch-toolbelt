import os
import warnings
from typing import Tuple, Optional, Mapping, Any

import numpy as np
import rasterio
from pytorch_toolbelt.utils import fs
from rasterio.windows import Window

__all__ = ["read_tiff", "ImageAnnotation"]


class ImageAnnotation:
    def __init__(self, row: Mapping[str, Any]):
        self.rows = row["rows"]
        self.cols = row["cols"]
        self.image_id = fs.id_from_fname(row["images"])
        self.image_path = row["images"]
        self.mask_path = row["masks"]
        self.location = row["location"]

    def load_image(self, crop_coords):
        image = read_tiff(self.image_path, crop_coords)
        mask = read_tiff(self.mask_path, crop_coords)
        return image, mask

    @property
    def shape(self):
        return self.rows, self.cols


def read_tiff(image_fname: str, crop_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> np.ndarray:
    window = None
    if crop_coords is not None:
        (row_start, row_stop), (col_start, col_stop) = crop_coords
        window = Window.from_slices((row_start, row_stop), (col_start, col_stop))

    if not os.path.isfile(image_fname):
        raise FileNotFoundError(image_fname)

    with warnings.catch_warnings():
        # warnings.simplefilter("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

        with rasterio.open(image_fname) as f:
            image = f.read(window=window)
            image = np.moveaxis(image, 0, -1)  # CHW->HWC
            if image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            elif image.shape[2] == 3:
                image = image
            else:
                warnings.warn(
                    f"Image contains unsupported number of channels {image.shape[2]}."
                    f"Only 1 or 3 channel TIFF images are supported. We will truncate image to the first 3 channels"
                )
                image = image[..., 0:3]
            return image
