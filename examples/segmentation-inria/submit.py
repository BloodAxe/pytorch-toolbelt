import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from catalyst.dl.utils import UtilsFactory
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.inference.tta import tta_d4_image2mask, tta_fliplr_image2mask
import albumentations as A
from pytorch_toolbelt.utils.fs import auto_file, find_in_dir, read_rgb_image
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from tqdm import tqdm

from models.factory import get_model


class TTAWrapperFlipLR(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return tta_d4_image2mask(self.model, x)


class TTAWrapperD4(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return tta_fliplr_image2mask(self.model, x)


class InMemoryDataset(Dataset):
    def __init__(self, data: List[Dict], transform: A.Compose):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(**self.data[item])


def _tensor_from_rgb_image(image: np.ndarray, **kwargs):
    return tensor_from_rgb_image(image)


def predict(model: nn.Module, image: np.ndarray, tta: str, image_size, normalize=A.Normalize(), batch_size=1, activation='sigmoid') -> np.ndarray:
    model.eval()
    tile_step = (image_size[0] // 2, image_size[1] // 2)

    w = np.ones(image_size, dtype=np.float32)
    tile_slicer = ImageSlicer(image.shape, image_size, tile_step, weight=w)
    tile_merger = CudaTileMerger(tile_slicer.target_shape, 1, w)
    patches = tile_slicer.split(image)

    if tta == 'fliplr':
        model = TTAWrapperFlipLR(model)
        print('Using FlipLR TTA')

    if tta == 'd4':
        model = TTAWrapperD4(model)
        print('Using FlipLR TTA', model.crop_size)

    transform = A.Compose([
        normalize,
        A.Lambda(image=_tensor_from_rgb_image)
    ])

    with torch.no_grad():
        data = list({'image': patch, 'coords': np.array(coords, dtype=np.int)} for (patch, coords) in zip(patches, tile_slicer.crops))
        for batch in DataLoader(InMemoryDataset(data, transform), pin_memory=True, batch_size=batch_size):
            image = batch['image'].cuda(non_blocking=True)
            coords = batch['coords']
            mask_batch = model(image)
            tile_merger.integrate_batch(mask_batch, coords)

    mask = tile_merger.merge()
    if activation == 'sigmoid':
        mask = mask.sigmoid()

    if isinstance(activation, float):
        mask = F.relu(mask_batch - activation, inplace=True)

    mask = np.moveaxis(to_numpy(mask), 0, -1)
    mask = tile_slicer.crop_to_orignal_size(mask)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data-dir', type=str, default=None, required=True, help='Data dir')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, required=True, help='Checkpoint filename to use as initial model weights')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('-tta', '--tta', default=None, type=str, help='Type of TTA to use [fliplr, d4]')
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint_file = auto_file(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_file))
    out_dir = os.path.join(run_dir, 'submit')
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = UtilsFactory.load_checkpoint(checkpoint_file)

    model = get_model('linknet152')
    UtilsFactory.unpack_checkpoint(checkpoint, model=model)

    model = model.cuda().eval()

    test_images = find_in_dir(os.path.join(data_dir, 'test', 'images'))
    for fname in tqdm(test_images, total=len(test_images)):
        image = read_rgb_image(fname)
        mask = predict(model, image, args.tta, image_size=(512, 512), batch_size=args.batch_size, activation='sigmoid')
        mask = ((mask > 0.5) * 255).astype(np.uint8)
        name = os.path.join(out_dir, os.path.basename(fname))
        cv2.imwrite(name, mask)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
