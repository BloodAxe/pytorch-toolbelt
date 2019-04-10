from typing import Dict, List
import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from pytorch_toolbelt.inference.tta import tta_d4_image2mask, tta_fliplr_image2mask
import albumentations as A
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F


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


def predict(model: nn.Module, image: np.ndarray, tta: str, image_size, normalize=A.Normalize(), batch_size=1, activation='sigmoid') -> np.ndarray:
    model.eval()
    tile_step = image_size / 2

    w = np.ones(image_size, dtype=np.float32)
    tile_slicer = ImageSlicer(image.shape, image_size, tile_step // 2, weight=w)
    tile_merger = CudaTileMerger(image.shape, 1, w)
    patches = tile_slicer.split(image)

    if tta == 'fliplr':
        model = TTAWrapperFlipLR(model)
        print('Using FlipLR TTA')

    if tta == 'd4':
        model = TTAWrapperD4(model)
        print('Using FlipLR TTA', model.crop_size)

    transform = A.Compose([
        normalize,
        A.Lambda(image=tensor_from_rgb_image)
    ])

    with torch.no_grad():
        for patch_batch, patch_coords in zip(DataLoader(InMemoryDataset(patches, transform), pin_memory=True, batch_size=batch_size), tile_slicer.crops):
            mask_batch = model(patch_batch)
            if activation == 'sigmoid':
                mask_batch = mask_batch.sigmoid()

            if isinstance(activation, float):
                mask_batch = F.relu(mask_batch - activation, inplace=True)

            tile_merger.integrate_batch(mask_batch, patch_coords)

    mask = tile_merger.merge()
    mask = to_numpy(mask.squeeze(0))
    return mask
