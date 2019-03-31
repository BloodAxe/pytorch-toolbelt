import collections

import cv2
import numpy as np
import torch
import albumentations as A
import torch.nn.functional as F
from catalyst.contrib.criterion import FocalLoss
from catalyst.dl.callbacks import EarlyStoppingCallback, PrecisionCallback, UtilsFactory, JaccardCallback
from catalyst.dl.experiments import SupervisedRunner
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pytorch_toolbelt.utils.fs import find_images_in_dir, read_rgb_image
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, tensor_from_rgb_image


def canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=300)
    return (edges > 0).astype(np.uint8)


class CannyModel(nn.Module):
    def __init__(self, input_channels=3, features=16):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.final = nn.Conv2d(features, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.selu(x)
        x = self.conv2(x)
        x = F.selu(x)
        x = self.final(x)
        return x


class EdgesDataset(Dataset):
    def __init__(self, images, image_size=(224, 224)):
        self.images = images
        self.transform = A.Compose([
            A.PadIfNeeded(224),
            A.RandomCrop(image_size[0], image_size[1]),
            A.RandomRotate90()
        ])
        self.normalize = A.Normalize()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = read_rgb_image(self.images[index])
        data = self.transform(image=image)
        data['mask'] = canny_edges(data['image'])
        data = self.normalize(**data)
        data['image'] = tensor_from_rgb_image(data['image'])
        data['mask'] = torch.from_numpy(data['mask']).float().unsqueeze(0)
        return {'features': data['image'], 'targets': data['mask']}


def main():
    images_dir = 'c:\datasets\ILSVRC2013_DET_val'

    canny_cnn = maybe_cuda(CannyModel())
    optimizer = Adam(canny_cnn.parameters(), lr=1e-4)

    images = find_images_in_dir(images_dir)
    train_images, valid_images = train_test_split(images, test_size=0.1, random_state=1234)

    num_workers = 4
    num_epochs = 100
    batch_size = 16

    train_loader = DataLoader(EdgesDataset(train_images), batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(EdgesDataset(valid_images), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=canny_cnn,
        criterion=FocalLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=[
            JaccardCallback(),
            EarlyStoppingCallback(patience=5, min_delta=0.01),
        ],
        loaders=loaders,
        logdir='logs',
        num_epochs=num_epochs,
        verbose=True,
        # check=True
    )

    UtilsFactory.plot_metrics(
        logdir='logs',
        metrics=["loss", "precision01", "precision03", "base/lr"])


if __name__ == '__main__':
    main()
