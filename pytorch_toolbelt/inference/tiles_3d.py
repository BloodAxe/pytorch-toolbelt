__all__ = ["VolumeSlicer", "VolumeMerger"]


class VolumeSlicer:
    """
    Helper class to slice 3d volume into smaller volumes and merge them back
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        voxel_size: Union[int, Tuple[int, int, int]],
        voxel_step: Union[int, Tuple[int, int, int]],
        weight="mean",
    ):
        """

        :param volume_shape: Shape of the source image (D, H, W, C)
        :param voxel_size: Tile size (Scalar or tuple (D, H, W, C)
        :param voxel_step: Step in pixels between voxels (Scalar or tuple (D, H, W))
        :param weight: Fusion algorithm. 'mean' - avergaing
        """
        self.volume_shape = np.array(volume_shape)[:3]

        if isinstance(voxel_size, (tuple, list)):
            if len(voxel_size) != 3:
                raise ValueError()
            self.tile_size = np.array(voxel_size, dtype=int)
        else:
            self.tile_size = np.array([int(voxel_size)] * 3)

        if isinstance(voxel_step, (tuple, list)):
            if len(voxel_step) != 3:
                raise ValueError()
            self.tile_step = np.array(voxel_step, dtype=int)
        else:
            self.tile_step = np.array([int(voxel_step)] * 3)

        self.weight = weight

        if self.tile_step[0] < 1 or self.tile_step[0] > self.tile_size[0]:
            raise ValueError()
        if self.tile_step[1] < 1 or self.tile_step[1] > self.tile_size[1]:
            raise ValueError()
        if self.tile_step[2] < 1 or self.tile_step[2] > self.tile_size[2]:
            raise ValueError()

        overlap = self.tile_size - self.tile_step

        # In case margin is not set, we compute it manually

        self.num_tiles = np.maximum(1, np.ceil((self.volume_shape - overlap) / self.tile_step)).astype(int)
        self.extra_pad = self.tile_step * self.num_tiles - (self.volume_shape - overlap)
        self.pad_before = self.extra_pad // 2
        self.pad_after = self.extra_pad - self.pad_before
        self.orignal_image_roi = (
            slice(self.pad_before[0], self.pad_before[0] + self.volume_shape[0]),
            slice(self.pad_before[1], self.pad_before[1] + self.volume_shape[1]),
            slice(self.pad_before[2], self.pad_before[2] + self.volume_shape[2]),
        )
        self.orignal_mask_roi = (
            slice(None),
            slice(self.pad_before[0], self.pad_before[0] + self.volume_shape[0]),
            slice(self.pad_before[1], self.pad_before[1] + self.volume_shape[1]),
            slice(self.pad_before[2], self.pad_before[2] + self.volume_shape[2]),
        )

        rois = []
        bbox_crops = []

        dim_i = range(0, self.volume_shape[0] + self.extra_pad[0] - self.tile_size[0] + 1, self.tile_step[0])
        dim_j = range(0, self.volume_shape[1] + self.extra_pad[1] - self.tile_size[1] + 1, self.tile_step[1])
        dim_k = range(0, self.volume_shape[2] + self.extra_pad[2] - self.tile_size[2] + 1, self.tile_step[2])

        for i in dim_i:
            for j in dim_j:
                for k in dim_k:
                    roi = (
                        slice(i, i + self.tile_size[0]),
                        slice(j, j + self.tile_size[1]),
                        slice(k, k + self.tile_size[2]),
                    )
                    roi2 = (
                        slice(i - self.pad_before[0], i + self.tile_size[0]),
                        slice(j - self.pad_before[1], j + self.tile_size[1]),
                        slice(k - self.pad_before[2], k + self.tile_size[2]),
                    )
                    rois.append(roi)
                    bbox_crops.append(roi2)

        self.crops = rois
        self.bbox_crops = bbox_crops

    def split(self, volume: np.ndarray, value=0) -> List[np.ndarray]:
        if (volume.shape != self.volume_shape).any():
            raise ValueError(f"Volume shape {volume.shape} is not equal to the expected {self.volume_shape}")

        pad_width = np.stack([self.pad_before, self.pad_after], axis=-1)
        image_pad = np.pad(volume, pad_width, mode="constant", constant_values=value)

        tiles = []
        for roi in self.crops:
            tile = image_pad[roi].copy()
            tiles.append(tile)

        return tiles

    def iter_split(self, volume, value=0) -> Iterable[Tuple[np.ndarray, Any]]:
        if (volume.shape != self.volume_shape).any():
            raise ValueError(f"Volume shape {volume.shape} is not equal to the expected {self.volume_shape}")

        pad_width = np.stack([self.pad_before, self.pad_after], axis=-1)
        image_pad = np.pad(volume, pad_width, mode="constant", constant_values=value)

        for roi in self.crops:
            tile = image_pad[roi].copy()
            yield tile, roi

    @property
    def target_shape(self):
        target_shape = self.volume_shape + self.extra_pad
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

    def crop_to_orignal_size(self, volume):
        return volume[self.orignal_image_roi]

    def _mean(self, volume_size):
        return np.ones(volume_size, dtype=np.float32)

    def _pyramid(self, tile_size):
        w, _, _ = compute_pyramid_patch_weight_loss(tile_size[0], tile_size[1])
        return w


class VolumeMerger:
    """
    Helper class to merge volume segmentations on CPU/GPU.
    """

    def __init__(self, volume_shape, channels: int, weight, device="cpu", dtype=torch.float32):
        """

        :param volume_shape: Shape of the source image
        :param weight: Weighting matrix
        """
        self.channels = channels

        self.weight = torch.from_numpy(np.expand_dims(weight, axis=0)).to(device=device, dtype=dtype)
        self.volume = torch.zeros((channels, *volume_shape), device=device, dtype=dtype)
        self.norm_mask = torch.zeros((1, *volume_shape), device=device, dtype=dtype)

    def accumulate_single(self, tile: torch.Tensor, roi):
        """
        Accumulates single element
        :param sample: Predicted image of shape [C,D,H,W]
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        tile = tile.to(device=self.volume.device)
        self.volume[:, roi] += tile * self.weight
        self.norm_mask[:, roi] += self.weight

    def integrate_batch(self, batch: torch.Tensor, rois):
        """
        Accumulates batch of tile predictions
        :param batch: Predicted tiles  of shape [B,C,D,H,W]
        :param crop_coords: Corresponding tile crops w.r.t to original image
        """
        if len(batch) != len(rois):
            raise ValueError("Number of images in batch does not correspond to number of coordinates")

        batch = batch.to(device=self.volume.device)
        for tile, roi in zip(batch, rois):
            weighted_tile = tile * self.weight
            self.volume[:, roi[0], roi[1], roi[2]] += weighted_tile.type_as(self.volume)
            self.norm_mask[:, roi[0], roi[1], roi[2]] += self.weight

    def merge(self) -> torch.Tensor:
        return self.volume / self.norm_mask
