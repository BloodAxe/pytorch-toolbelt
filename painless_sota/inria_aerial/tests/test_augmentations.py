import os.path

import cv2
from hydra.utils import instantiate
from omegaconf import OmegaConf

import albumentations as A
from skimage.measure import label
from skimage.segmentation import find_boundaries

from painless_sota.inria_aerial.data.image_io import read_tiff
from painless_sota.inria_aerial.hydra_utils import register_albumentations_resolver
from pytorch_toolbelt.utils import fs


def test_dataset():
    images = fs.find_images_in_dir("C:/datasets/inria/AerialImageDataset/train/images")
    masks = fs.find_images_in_dir("C:/datasets/inria/AerialImageDataset/train/gt")

    for image_path, mask in zip(images, masks):
        boundaries = find_boundaries(label(read_tiff(mask)))
        image = read_tiff(image_path)
        image2 = image.copy()
        image2[boundaries] = (255, 0, 255)
        image = cv2.addWeighted(image, 0.5, image2, 0.5, 0)
        cv2.imwrite(filename=fs.change_extension(os.path.basename(image_path), ".jpg"), img=image[..., ::-1])


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
