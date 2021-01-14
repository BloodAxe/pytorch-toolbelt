"""Implementation of tile-based inference allowing to predict huge images that does not fit into GPU memory entirely
in a sliding-window fashion and merging prediction mask back to full-resolution.
"""
import math
from typing import List, Iterable

import cv2
import numpy as np
import torch

from pytorch_toolbelt.utils import pytorch_toolbelt_deprecated

__all__ = ["ImageSlicer", "TileMerger", "CudaTileMerger", "compute_pyramid_patch_weight_loss"]


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix that assigns bigger weight on pixels in center and
    less weight to pixels on image boundary.
    This weight matrix then used for merging individual tile predictions and helps dealing
    with prediction artifacts on tile boundaries.

    :param width: Tile width
    :param height: Tile height
    :return: Since-channel image [Width x Height]
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W, Dc, De


class ImageSlicer:
    """
    Helper class to slice image into tiles and merge them back
    """

    def __init__(self, image_shape, tile_size, tile_step=0, image_margin=0, weight="mean"):
        """

        :param image_shape: Shape of the source image (H, W)
        :param tile_size: Tile size (Scalar or tuple (H, W)
        :param tile_step: Step in pixels between tiles (Scalar or tuple (H, W))
        :param image_margin:
        :param weight: Fusion algorithm. 'mean' - avergaing
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]

        if isinstance(tile_size, (tuple, list)):
            assert len(tile_size) == 2
            self.tile_size = int(tile_size[0]), int(tile_size[1])
        else:
            self.tile_size = int(tile_size), int(tile_size)

        if isinstance(tile_step, (tuple, list)):
            assert len(tile_step) == 2
            self.tile_step = int(tile_step[0]), int(tile_step[1])
        else:
            self.tile_step = int(tile_step), int(tile_step)

        weights = {"mean": self._mean, "pyramid": self._pyramid}

        self.weight = weight if isinstance(weight, np.ndarray) else weights[weight](self.tile_size)

        if self.tile_step[0] < 1 or self.tile_step[0] > self.tile_size[0]:
            raise ValueError()
        if self.tile_step[1] < 1 or self.tile_step[1] > self.tile_size[1]:
            raise ValueError()

        overlap = [self.tile_size[0] - self.tile_step[0], self.tile_size[1] - self.tile_step[1]]

        self.margin_left = 0
        self.margin_right = 0
        self.margin_top = 0
        self.margin_bottom = 0

        if image_margin == 0:
            # In case margin is not set, we compute it manually

            nw = max(1, math.ceil((self.image_width - overlap[1]) / self.tile_step[1]))
            nh = max(1, math.ceil((self.image_height - overlap[0]) / self.tile_step[0]))

            extra_w = self.tile_step[1] * nw - (self.image_width - overlap[1])
            extra_h = self.tile_step[0] * nh - (self.image_height - overlap[0])

            self.margin_left = extra_w // 2
            self.margin_right = extra_w - self.margin_left
            self.margin_top = extra_h // 2
            self.margin_bottom = extra_h - self.margin_top

        else:
            if (self.image_width - overlap[1] + 2 * image_margin) % self.tile_step[1] != 0:
                raise ValueError()

            if (self.image_height - overlap[0] + 2 * image_margin) % self.tile_step[0] != 0:
                raise ValueError()

            self.margin_left = image_margin
            self.margin_right = image_margin
            self.margin_top = image_margin
            self.margin_bottom = image_margin

        crops = []
        bbox_crops = []

        for y in range(
            0, self.image_height + self.margin_top + self.margin_bottom - self.tile_size[0] + 1, self.tile_step[0]
        ):
            for x in range(
                0, self.image_width + self.margin_left + self.margin_right - self.tile_size[1] + 1, self.tile_step[1]
            ):
                crops.append((x, y, self.tile_size[1], self.tile_size[0]))
                bbox_crops.append((x - self.margin_left, y - self.margin_top, self.tile_size[1], self.tile_size[0]))

        self.crops = np.array(crops)
        self.bbox_crops = np.array(bbox_crops)

    def iter_split(self, image: np.ndarray, border_type=cv2.BORDER_CONSTANT, value=0) -> Iterable:
        if (image.shape[0] != self.image_height) or (image.shape[1] != self.image_width):
            raise ValueError()

        orig_shape_len = len(image.shape)

        for coords, crop_coords in zip(self.crops, self.bbox_crops):
            x, y, tile_width, tile_height = crop_coords
            x1 = max(x, 0)
            y1 = max(y, 0)
            x2 = min(image.shape[1], x + tile_width)
            y2 = min(image.shape[0], y + tile_height)

            tile = image[y1:y2, x1:x2]  # .copy()
            if x < 0 or y < 0 or (x + tile_width) > image.shape[1] or (y + tile_height) > image.shape[0]:
                tile = cv2.copyMakeBorder(
                    tile,
                    max(0, -y),
                    max(0, y + tile_height - image.shape[0]),
                    max(0, -x),
                    max(0, x + tile_width - image.shape[1]),
                    borderType=border_type,
                    value=value,
                )

                # This check recovers possible lack of last dummy dimension for single-channel images
                if len(tile.shape) != orig_shape_len:
                    tile = np.expand_dims(tile, axis=-1)

            yield tile, coords

    def split(self, image, border_type=cv2.BORDER_CONSTANT, value=0):
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        orig_shape_len = len(image.shape)
        image = cv2.copyMakeBorder(
            image,
            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,
            borderType=border_type,
            value=value,
        )

        # This check recovers possible lack of last dummy dimension for single-channel images
        if len(image.shape) != orig_shape_len:
            image = np.expand_dims(image, axis=-1)

        tiles = []
        for x, y, tile_width, tile_height in self.crops:
            tile = image[y : y + tile_height, x : x + tile_width]  # .copy()
            assert tile.shape[0] == self.tile_size[0]
            assert tile.shape[1] == self.tile_size[1]

            tiles.append(tile)

        return tiles

    def cut_patch(self, image: np.ndarray, slice_index, border_type=cv2.BORDER_CONSTANT, value=0):
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        orig_shape_len = len(image.shape)
        image = cv2.copyMakeBorder(
            image,
            self.margin_top,
            self.margin_bottom,
            self.margin_left,
            self.margin_right,
            borderType=border_type,
            value=value,
        )

        # This check recovers possible lack of last dummy dimension for single-channel images
        if len(image.shape) != orig_shape_len:
            image = np.expand_dims(image, axis=-1)

        x, y, tile_width, tile_height = self.crops[slice_index]

        tile = image[y : y + tile_height, x : x + tile_width].copy()
        assert tile.shape[0] == self.tile_size[0]
        assert tile.shape[1] == self.tile_size[1]
        return tile

    @property
    def target_shape(self):
        target_shape = (
            self.image_height + self.margin_bottom + self.margin_top,
            self.image_width + self.margin_right + self.margin_left,
        )
        return target_shape

    def merge(self, tiles: List[np.ndarray], dtype=np.float32):
        if len(tiles) != len(self.crops):
            raise ValueError

        channels = 1 if len(tiles[0].shape) == 2 else tiles[0].shape[2]
        target_shape = (
            self.image_height + self.margin_bottom + self.margin_top,
            self.image_width + self.margin_right + self.margin_left,
            channels,
        )

        image = np.zeros(target_shape, dtype=np.float64)
        norm_mask = np.zeros(target_shape, dtype=np.float64)

        w = np.dstack([self.weight] * channels)

        for tile, (x, y, tile_width, tile_height) in zip(tiles, self.crops):
            # print(x, y, tile_width, tile_height, image.shape)
            image[y : y + tile_height, x : x + tile_width] += tile * w
            norm_mask[y : y + tile_height, x : x + tile_width] += w

        # print(norm_mask.min(), norm_mask.max())
        norm_mask = np.clip(norm_mask, a_min=np.finfo(norm_mask.dtype).eps, a_max=None)
        normalized = np.divide(image, norm_mask).astype(dtype)
        crop = self.crop_to_orignal_size(normalized)
        return crop

    def crop_to_orignal_size(self, image):
        assert image.shape[0] == self.target_shape[0]
        assert image.shape[1] == self.target_shape[1]
        crop = image[
            self.margin_top : self.image_height + self.margin_top,
            self.margin_left : self.image_width + self.margin_left,
        ]
        assert crop.shape[0] == self.image_height
        assert crop.shape[1] == self.image_width
        return crop

    def _mean(self, tile_size):
        return np.ones((tile_size[0], tile_size[1]), dtype=np.float32)

    def _pyramid(self, tile_size):
        w, _, _ = compute_pyramid_patch_weight_loss(tile_size[0], tile_size[1])
        return w


class TileMerger:
    """
    Helper class to merge final image on GPU. This generally faster than moving individual tiles to CPU.
    """

    def __init__(self, image_shape, channels, weight, device="cpu", dtype=torch.float32):
        """

        :param image_shape: Shape of the source image
        :param image_margin:
        :param weight: Weighting matrix
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.channels = channels

        self.weight = torch.from_numpy(np.expand_dims(weight, axis=0)).to(device=device, dtype=dtype)
        self.image = torch.zeros((channels, self.image_height, self.image_width), device=device, dtype=dtype)
        self.norm_mask = torch.zeros((1, self.image_height, self.image_width), device=device, dtype=dtype)

    def accumulate_single(self, tile: torch.Tensor, coords):
        """
        Accumulates single element
        :param sample: Predicted image of shape [C,H,W]
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        tile = tile.to(device=self.image.device)
        x, y, tile_width, tile_height = coords
        self.image[:, y : y + tile_height, x : x + tile_width] += tile * self.weight
        self.norm_mask[:, y : y + tile_height, x : x + tile_width] += self.weight

    def integrate_batch(self, batch: torch.Tensor, crop_coords):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles  of shape [B,C,H,W]
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(crop_coords):
            raise ValueError("Number of images in batch does not correspond to number of coordinates")

        batch = batch.to(device=self.image.device)
        for tile, (x, y, tile_width, tile_height) in zip(batch, crop_coords):
            self.image[:, y : y + tile_height, x : x + tile_width] += tile * self.weight
            self.norm_mask[:, y : y + tile_height, x : x + tile_width] += self.weight

    def merge(self) -> torch.Tensor:
        return self.image / self.norm_mask


@pytorch_toolbelt_deprecated("This class is deprecated and will be removed in 0.5.0. Please use TileMerger instead.")
class CudaTileMerger(TileMerger):
    def __init__(self, image_shape, channels, weight, device="cuda"):
        super().__init__(image_shape, channels, weight, device)
