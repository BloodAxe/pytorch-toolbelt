import numpy as np
import torch

from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, rgb_image_from_tensor, to_numpy


def test_tiles_split_merge():
    image = np.random.random((5000, 5000, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=512, tile_step=256, weight='mean')
    tiles = tiler.split(image)
    merged = tiler.merge(tiles, dtype=np.uint8)
    np.testing.assert_equal(merged, image)


def test_tiles_split_merge_2():
    image = np.random.random((5000, 5000, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256), weight='pyramid')

    np.testing.assert_allclose(tiler.weight, tiler.weight.T)

    tiles = tiler.split(image)
    merged = tiler.merge(tiles, dtype=np.uint8)
    np.testing.assert_equal(merged, image)


def test_tiles_split_merge_cuda():
    if not torch.cuda.is_available():
        return
    image = np.random.random((5000, 5000, 3)).astype(np.uint8)
    tiler = ImageSlicer(image.shape, tile_size=(512, 512), tile_step=(256, 256), weight='pyramid')
    tiles = tiler.split(image)

    merger = CudaTileMerger(tiler.target_shape, 3, tiler.weight)
    for tile, coord in zip(tiles, tiler.crops):
        batch = tensor_from_rgb_image(tile).unsqueeze(0).float().cuda()
        merger.integrate_batch(batch, [coord])

    merged = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
    merged = tiler.crop_to_orignal_size(merged)

    np.testing.assert_equal(merged, image)
