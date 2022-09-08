import os.path

import cv2
from hydra.utils import instantiate
from omegaconf import OmegaConf

import albumentations as A

from painless_sota.inria_aerial.data.image_io import read_tiff
from painless_sota.inria_aerial.hydra_utils import register_albumentations_resolver

A.Perspective


def test_augmentations():
    register_albumentations_resolver()
    config = OmegaConf.load("../configs/augmentations/hard.yaml")
    augmentations = instantiate(config)

    pipeline = A.Compose([A.CenterCrop(1024, 1024), *augmentations])

    image = read_tiff("C:/datasets/inria/AerialImageDataset/train/images/austin1.tif")
    mask = read_tiff("C:/datasets/inria/AerialImageDataset/train/gt/austin1.tif")

    for i in range(64):
        output = pipeline(image=image, mask=mask)["image"]
        cv2.imwrite(f"augmentation_{i:04d}.jpg", output[..., ::-1])
