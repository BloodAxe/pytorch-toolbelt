from typing import Callable, List

from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, tensor_from_mask_image
from torch.utils.data import Dataset, ConcatDataset


class ImageMaskDataset(Dataset):
    def __init__(self, image_filenames, target_filenames, image_loader, target_loader, transform=None, keep_in_mem=False):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        self.image_ids = [id_from_fname(fname) for fname in image_filenames]

        if keep_in_mem:
            self.images = [image_loader(fname) for fname in image_filenames]
            self.masks = [target_loader(fname) for fname in target_filenames]
            self.get_image = lambda x: x
            self.get_loader = lambda x: x
        else:
            self.images = image_filenames
            self.masks = target_filenames
            self.get_image = image_loader
            self.get_loader = target_loader

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.get_image(self.images[index])
        mask = self.get_loader(self.masks[index])

        data = self.transform(image=image, mask=mask)

        return {'features': tensor_from_rgb_image(data['image']),
                'targets': tensor_from_mask_image(data['mask']).float(),
                'image_id': self.image_ids[index]}


class TiledSingleImageDataset(Dataset):
    def __init__(self, image_fname: str,
                 mask_fname: str,
                 image_loader: Callable,
                 target_loader: Callable,
                 tile_size,
                 tile_step,
                 image_margin=0,
                 transform=None,
                 target_shape=None,
                 keep_in_mem=False):
        self.image_fname = image_fname
        self.mask_fname = mask_fname
        self.image_loader = image_loader
        self.mask_loader = target_loader
        self.image = None
        self.mask = None

        if target_shape is None or keep_in_mem:
            image = image_loader(image_fname)
            mask = target_loader(mask_fname)
            if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
                raise ValueError(f"Image size {image.shape} and mask shape {image.shape} must have equal width and height")

            target_shape = image.shape

        self.slicer = ImageSlicer(target_shape, tile_size, tile_step, image_margin)

        if keep_in_mem:
            self.images = self.slicer.split(image)
            self.masks = self.slicer.split(mask)
        else:
            self.images = None
            self.masks = None

        self.transform = transform
        self.image_ids = [id_from_fname(image_fname) + f' [{crop[0]};{crop[1]};{crop[2]};{crop[3]};]' for crop in self.slicer.crops]

    def _get_image(self, index):
        if self.images is None:
            image = self.image_loader(self.image_fname)
            image = self.slicer.cut_patch(image, index)
        else:
            image = self.images[index]
        return image

    def _get_mask(self, index):
        if self.masks is None:
            mask = self.mask_loader(self.mask_fname)
            mask = self.slicer.cut_patch(mask, index)
        else:
            mask = self.masks[index]
        return mask

    def __len__(self):
        return len(self.slicer.crops)

    def __getitem__(self, index):
        image = self._get_image(index)
        mask = self._get_mask(index)
        data = self.transform(image=image, mask=mask)

        return {'features': tensor_from_rgb_image(data['image']),
                'targets': tensor_from_mask_image(data['mask']).float(),
                'image_id': self.image_ids[index]}


class TiledImageMaskDataset(ConcatDataset):
    def __init__(self,
                 image_filenames: List[str],
                 target_filenames: List[str],
                 image_loader: Callable,
                 target_loader: Callable,
                 **kwargs):
        if len(image_filenames) != len(target_filenames):
            raise ValueError('Number of images does not corresponds to number of targets')

        datasets = []
        for image, mask in zip(image_filenames, target_filenames):
            dataset = TiledSingleImageDataset(image, mask, image_loader, target_loader, **kwargs)
            datasets.append(dataset)
        super().__init__(datasets)
