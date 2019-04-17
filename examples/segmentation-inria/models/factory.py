import os
from functools import partial
from multiprocessing.pool import Pool
from typing import List, Dict

import cv2
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F
from tqdm import tqdm

from pytorch_toolbelt.inference.tiles import CudaTileMerger, ImageSlicer
from pytorch_toolbelt.inference.tta import tta_fliplr_image2mask, tta_d4_image2mask
from pytorch_toolbelt.losses.focal import BinaryFocalLoss
from pytorch_toolbelt.losses.jaccard import BinaryJaccardLogLoss
from pytorch_toolbelt.losses.lovasz import BinaryLovaszLoss
from pytorch_toolbelt.losses.joint_loss import JointLoss
from pytorch_toolbelt.utils.dataset_utils import TiledImageMaskDataset, ImageMaskDataset, TiledSingleImageDataset
from pytorch_toolbelt.utils.fs import read_rgb_image, read_image_as_is
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy, rgb_image_from_tensor

import numpy as np
import albumentations as A

from .fpn import fpn128_resnext50, fpn256_resnext50, fpn128_resnet34
from .linknet import LinkNet152, LinkNet34
from .unet import UNet


def get_model(model_name: str, image_size=None) -> nn.Module:
    registry = {
        'unet': partial(UNet, upsample=False),
        'linknet34': LinkNet34,
        'linknet152': LinkNet152,
        'fpn128_resnet34': fpn128_resnet34,
        'fpn128_resnext50': fpn128_resnext50,
        'fpn256_resnext50': fpn256_resnext50
    }

    return registry[model_name.lower()]()


def get_optimizer(optimizer_name: str, parameters, lr: float, **kwargs):
    from torch.optim import SGD, Adam

    if optimizer_name.lower() == 'sgd':
        return SGD(parameters, lr, momentum=0.9, nesterov=True, **kwargs)

    if optimizer_name.lower() == 'adam':
        return Adam(parameters, lr, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)


def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == 'bce':
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == 'focal':
        return BinaryFocalLoss(alpha=None, gamma=1.5, **kwargs)

    if loss_name.lower() == 'bce_jaccard':
        return JointLoss(first=BCEWithLogitsLoss(), second=BinaryJaccardLogLoss(), first_weight=1.0, second_weight=0.5)

    if loss_name.lower() == 'bce_lovasz':
        return JointLoss(first=BCEWithLogitsLoss(), second=BinaryLovaszLoss(), first_weight=1.0, second_weight=0.5)

    raise KeyError(loss_name)


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


def predict(model: nn.Module, image: np.ndarray, image_size, tta=None, normalize=A.Normalize(), batch_size=1, activation='sigmoid') -> np.ndarray:
    model.eval()
    tile_step = (image_size[0] // 2, image_size[1] // 2)

    tile_slicer = ImageSlicer(image.shape, image_size, tile_step, weight='pyramid')
    tile_merger = CudaTileMerger(tile_slicer.target_shape, 1, tile_slicer.weight)
    patches = tile_slicer.split(image)

    transform = A.Compose([
        normalize,
        A.Lambda(image=_tensor_from_rgb_image)
    ])

    if tta == 'fliplr':
        model = TTAWrapperFlipLR(model)
        print('Using FlipLR TTA')

    if tta == 'd4':
        model = TTAWrapperD4(model)
        print('Using D4 TTA')

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


def __compute_ious(args):
    thresholds = np.arange(0, 256)
    gt, pred = args
    gt = cv2.imread(gt) > 0  # Make binary {0,1}
    pred = cv2.imread(pred)

    pred_i = np.zeros_like(gt)

    intersection = np.zeros(len(thresholds))
    union = np.zeros(len(thresholds))

    gt_sum = gt.sum()
    for index, threshold in enumerate(thresholds):
        np.greater(pred, threshold, out=pred_i)
        union[index] += gt_sum + pred_i.sum()

        np.logical_and(gt, pred_i, out=pred_i)
        intersection[index] += pred_i.sum()

    return intersection, union


def optimize_threshold(gt_images, pred_images):
    thresholds = np.arange(0, 256)

    intersection = np.zeros(len(thresholds))
    union = np.zeros(len(thresholds))

    with Pool(32) as wp:
        for i, u in tqdm(wp.imap_unordered(__compute_ious, zip(gt_images, pred_images)), total=len(gt_images)):
            intersection += i
            union += u

    return thresholds, intersection / (union - intersection)


def read_inria_mask(fname):
    mask = read_image_as_is(fname)
    return (mask > 0).astype(np.uint8)


def get_dataloaders(data_dir: str,
                    batch_size=16,
                    num_workers=4,
                    fast=False,
                    image_size=(224, 224),
                    use_d4=True):
    locations = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

    train_data = []
    valid_data = []

    # For validation, we suggest to remove the first five images of every location (e.g., austin{1-5}.tif, chicago{1-5}.tif) from the training set.
    if fast:
        for loc in locations:
            valid_data.append(f'{loc}1')
            train_data.append(f'{loc}6')
    else:
        for loc in locations:
            for i in range(1, 6):
                valid_data.append(f'{loc}{i}')
            for i in range(6, 37):
                train_data.append(f'{loc}{i}')

    train_img = [os.path.join(data_dir, 'train', 'images', f'{fname}.tif') for fname in train_data]
    valid_img = [os.path.join(data_dir, 'train', 'images', f'{fname}.tif') for fname in valid_data]

    train_mask = [os.path.join(data_dir, 'train', 'gt', f'{fname}.tif') for fname in train_data]
    valid_mask = [os.path.join(data_dir, 'train', 'gt', f'{fname}.tif') for fname in valid_data]

    train_transform = A.Compose([
        # Make random-sized crop with scale [50%..200%] of target size 1.5 larger than target crop to have some space around for
        # further transforms
        A.RandomSizedCrop((image_size[0] // 2, image_size[1] * 2), int(image_size[0] * 1.5), int(image_size[1] * 1.5)),

        # Apply random rotations
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(alpha_affine=0, border_mode=cv2.BORDER_CONSTANT),
        ]),

        # Add occasion blur/sharpening
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
            A.IAASharpen()
        ]),

        # Crop to desired image size
        A.CenterCrop(image_size[0], image_size[1]),

        # D4 Augmentations
        A.Compose([
            A.Transpose(),
            A.RandomRotate90(),
        ], p=float(use_d4)),
        # In case we don't want to use D4 augmentations, we use flips
        A.HorizontalFlip(p=float(not use_d4)),

        # Spatial-preserving augmentations:
        A.OneOf([
            A.Cutout(),
            A.GaussNoise(),
        ]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.CLAHE(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma()
        ]),
        # Weather effects
        # A.OneOf([
        #     A.RandomFog(),
        #     A.RandomRain(),
        #     A.RandomSunFlare()
        # ]),

        # Normalize image to make use of pretrained model
        A.Normalize()
    ])

    if fast:
        trainset = TiledSingleImageDataset(train_img[0], train_mask[0], read_rgb_image, read_inria_mask,
                                           transform=train_transform,
                                           tile_size=(int(image_size[0] * 2), int(image_size[1] * 2)),
                                           tile_step=image_size,
                                           keep_in_mem=True)

        validset = TiledSingleImageDataset(valid_img[0], valid_mask[0], read_rgb_image, read_inria_mask,
                                           transform=A.Normalize(),
                                           # For validation we don't want tiles overlap
                                           tile_size=image_size,
                                           tile_step=image_size,
                                           keep_in_mem=True)

        num_train_samples = int(len(trainset) * (5000 * 5000) / (image_size[0] * image_size[1]))
        train_sampler = WeightedRandomSampler(np.ones(len(trainset)), num_train_samples)
    else:
        trainset = ImageMaskDataset(train_img, train_mask, read_rgb_image, read_inria_mask,
                                    transform=train_transform,
                                    keep_in_mem=False)

        validset = TiledImageMaskDataset(valid_img, valid_mask, read_rgb_image, read_inria_mask,
                                         transform=A.Normalize(),
                                         # For validation we don't want tiles overlap
                                         tile_size=image_size,
                                         tile_step=image_size,
                                         target_shape=(5000, 5000),
                                         keep_in_mem=False)

        num_train_samples = int(len(trainset) * (5000 * 5000) / (image_size[0] * image_size[1]))
        train_sampler = WeightedRandomSampler(np.ones(len(trainset)), num_train_samples)

    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True,
                             shuffle=train_sampler is None,
                             sampler=train_sampler)

    validloader = DataLoader(validset,
                             batch_size=batch_size,
                             num_workers=0 if fast else num_workers,
                             pin_memory=True,
                             shuffle=False)

    return trainloader, validloader


def visualize_inria_predictions(input: dict, output: dict, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    images = []
    for image, target, image_id, logits in zip(input['features'], input['targets'], input['image_id'], output['logits']):
        image = rgb_image_from_tensor(image, mean, std)
        target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits).squeeze(0)

        overlay = np.zeros_like(image)
        true_mask = target > 0
        pred_mask = logits > 0

        overlay[true_mask & pred_mask] = np.array([0, 250, 0], dtype=overlay.dtype)  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array([250, 250, 0], dtype=overlay.dtype)  # False alarm painted with yellow

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images
