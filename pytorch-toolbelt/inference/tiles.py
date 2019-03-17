"""Implementation of tile-based inference allowing to predict huge images that does not fit into GPU memory entirely
in a sliding-window fashion and merging prediction mask back to full-resolution.
"""

import numpy as np
import cv2
import math
import torch


def compute_pyramid_patch_weight_loss(width, height) -> np.ndarray:
    """Computes a weight image that puts more importance on center pixels of an image and
    puts less weight to pixels on image boundary.
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

    for i in range(width):
        for j in range(height):
            Dc[i, j] = np.sqrt(np.square(i - xc + 0.5) + np.square(j - yc + 0.5))
            De_l = np.sqrt(np.square(i - xl + 0.5) + np.square(j - j + 0.5))
            De_r = np.sqrt(np.square(i - xr + 0.5) + np.square(j - j + 0.5))
            De_b = np.sqrt(np.square(i - i + 0.5) + np.square(j - yb + 0.5))
            De_t = np.sqrt(np.square(i - i + 0.5) + np.square(j - yt + 0.5))
            De[i, j] = np.min([De_l, De_r, De_b, De_t])

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W, Dc, De


class ImageSlicer:
    """
    Helper class to slice image into tiles and merge them back
    """

    def __init__(self, image_shape, tile_size, tile_step=0, image_margin=0, weight='mean'):
        """

        :param image_shape: Shape of the source image
        :param tile_size: Tile size (Scalar or tuple (height, width)
        :param tile_step: Step in pixels between tiles (Scalar or tuple (height, width)
        :param image_margin:
        :param weight: Fusion algorithm. 'mean' - avergaing
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.tile_size = (tile_size, tile_size) if isinstance(tile_size, int) else int(tile_size[0]), int(tile_size[1])
        self.tile_step = (tile_step, tile_step) if isinstance(tile_step, int) else int(tile_step[0]), int(tile_step[1])

        weights = {
            'mean': self._mean,
            'pyramid': self._pyramid
        }

        self.compute_weight = weights[weight]

        if self.tile_step[0] < 1 or self.tile_step[0] > self.tile_size[0]:
            raise ValueError()
        if self.tile_step[1] < 1 or self.tile_step[1] > self.tile_size[1]:
            raise ValueError()

        overlap = [
            self.tile_size[0] - self.tile_step[0],
            self.tile_size[1] - self.tile_step[1],
        ]

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

        self.crops = []
        self.bbox_crops = []

        for y in range(0, self.image_height + self.margin_top + self.margin_bottom - self.tile_size[0] + 1, self.tile_step[0]):
            for x in range(0, self.image_width + self.margin_left + self.margin_right - self.tile_size[1] + 1, self.tile_step[1]):
                self.crops.append((x, y, self.tile_size[1], self.tile_size[0]))
                self.bbox_crops.append((x - self.margin_left, y - self.margin_top, self.tile_size[1], self.tile_size[0]))

    def split(self, image, borderType=cv2.BORDER_REFLECT101, value=0):
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        orig_shape_len = len(image.shape)
        image = cv2.copyMakeBorder(image, self.margin_top, self.margin_bottom, self.margin_left, self.margin_right, borderType=borderType, value=value)

        # This check recovers possible lack of last dummy dimension for single-channel images
        if len(image.shape) != orig_shape_len:
            image = np.expand_dims(image, axis=-1)

        tiles = []
        for x, y, tile_width, tile_height in self.crops:
            tile = image[y:y + tile_height, x:x + tile_width].copy()
            assert tile.shape[0] == self.tile_size[0]
            assert tile.shape[1] == self.tile_size[1]

            tiles.append(tile)

        return tiles

    def cut_patch(self, image, slice_index, borderType=cv2.BORDER_REFLECT101, value=0):
        assert image.shape[0] == self.image_height
        assert image.shape[1] == self.image_width

        orig_shape_len = len(image.shape)
        image = cv2.copyMakeBorder(image, self.margin_top, self.margin_bottom, self.margin_left, self.margin_right, borderType=borderType, value=value)

        # This check recovers possible lack of last dummy dimension for single-channel images
        if len(image.shape) != orig_shape_len:
            image = np.expand_dims(image, axis=-1)

        x, y, tile_width, tile_height = self.crops[slice_index]

        tile = image[y:y + tile_height, x:x + tile_width].copy()
        assert tile.shape[0] == self.tile_size[0]
        assert tile.shape[1] == self.tile_size[1]
        return tile

    def merge(self, tiles, dtype=np.float32):
        if len(tiles) != len(self.crops):
            raise ValueError

        channels = 1 if len(tiles[0].shape) == 2 else tiles[0].shape[2]
        target_shape = self.image_height + self.margin_bottom + self.margin_top, self.image_width + self.margin_right + self.margin_left, channels

        image = np.zeros(target_shape, dtype=np.float64)
        norm_mask = np.zeros(target_shape, dtype=np.float64)

        weight = self.compute_weight(self.tile_size)
        w = np.dstack([weight] * channels)

        for tile, (x, y, tile_width, tile_height) in zip(tiles, self.crops):
            # print(x, y, tile_width, tile_height, image.shape)
            image[y:y + tile_height, x:x + tile_width] += tile * w
            norm_mask[y:y + tile_height, x:x + tile_width] += w

        # print(norm_mask.min(), norm_mask.max())
        norm_mask = np.clip(norm_mask, a_min=np.finfo(norm_mask.dtype).eps, a_max=None)
        normalized = np.divide(image, norm_mask).astype(dtype)
        crop = normalized[self.margin_top:self.image_height + self.margin_top, self.margin_left:self.image_width + self.margin_left]
        assert crop.shape[0] == self.image_height
        assert crop.shape[1] == self.image_width
        return crop

    def _mean(self, tile_size):
        return np.ones((tile_size[0], tile_size[1]), dtype=np.float32)

    def _pyramid(self, tile_size):
        w, _, _ = compute_pyramid_patch_weight_loss(tile_size[0], tile_size[1])
        return w


class CudaTileMerger:
    """
    Helper class to merge final image on GPU. This generally faster than moving individual tiles to CPU.
    """

    def __init__(self, image_shape, channels, weight):
        """

        :param image_shape: Shape of the source image
        :param image_margin:
        :param weight: Weighting matrix
        """
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]

        self.weight = torch.from_numpy(np.expand_dims(weight, axis=0)).float().cuda()
        self.channels = channels
        self.image = torch.zeros((channels, self.image_height, self.image_width)).cuda()
        self.norm_mask = torch.zeros((1, self.image_height, self.image_width)).cuda()

    def integrate_batch(self, batch, crop_coords):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(crop_coords):
            raise ValueError

        for tile, (x, batch, tile_width, tile_height) in zip(batch, crop_coords):
            self.image[:, batch:batch + tile_height, x:x + tile_width] += tile * self.weight
            self.norm_mask[:, batch:batch + tile_height, x:x + tile_width] += self.weight

    def merge(self):
        self.image /= self.norm_mask
        return self.image
