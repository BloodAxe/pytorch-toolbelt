import collections

import albumentations as A
import cv2
import numpy as np
import torch
from catalyst.contrib.criterion import FocalLoss
from catalyst.dl.callbacks import EarlyStoppingCallback, JaccardCallback, Callback, \
    RunnerState, TensorboardLogger
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory
from sklearn.model_selection import train_test_split
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pytorch_toolbelt.utils.fs import find_images_in_dir, read_rgb_image
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, tensor_from_rgb_image, rgb_image_from_tensor, to_numpy
from pytorch_toolbelt.utils.catalyst_utils import EpochJaccardMetric, ShowPolarBatchesCallback


def canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=300)
    return (edges > 0).astype(np.uint8)


def conv_bn_relu(input_channels, output_channels):
    return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(output_channels),
                         nn.LeakyReLU(inplace=True))


class PoolUnpool(nn.Module):
    def __init__(self, kernel_size=3, padding=1, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, padding=1, stride=stride, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        x, index = self.pool(x)
        unpuul = self.unpool(x, index)
        return unpuul


class PolarGradients(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.gray = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.smooth = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.dx = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.dy = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        gaussian_kernel = np.array([[1, 4, 7, 4, 1],
                                    [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7],
                                    [4, 16, 26, 16, 4],
                                    [1, 4, 7, 4, 1]], dtype=np.float32) / 273.

        dx_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]], dtype=np.float32)

        dy_kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]], dtype=np.float32)

        with torch.no_grad():
            self.smooth.weight.set_(torch.from_numpy(gaussian_kernel).unsqueeze(0).unsqueeze(0))
            self.dx.weight.set_(torch.from_numpy(dx_kernel).unsqueeze(0).unsqueeze(0))
            self.dy.weight.set_(torch.from_numpy(dy_kernel).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        x = self.gray(x)
        x = self.smooth(x)
        dx = self.dx(x)
        dy = self.dy(x)
        mag = torch.sqrt(dx * dx + dy * dy + 1e-4)
        ori = torch.atan2(dy, dx)
        return torch.cat([mag, ori], dim=1)


class CannyModel(nn.Module):
    def __init__(self, input_channels=3, features=16):
        super().__init__()
        # self.polar_grad = PolarGradients(input_channels)
        self.conv1 = conv_bn_relu(input_channels, features)
        self.conv2 = conv_bn_relu(features, features)

        self.conv3 = conv_bn_relu(features * 2, features)
        self.conv4 = conv_bn_relu(features * 2, features)

        self.pool1 = PoolUnpool(features)
        self.pool2 = PoolUnpool(features)

        self.final = nn.Conv2d(features, 1, kernel_size=1)

    def forward(self, x):
        # pg = self.polar_grad(x)
        x = self.conv1(x)
        x = self.conv2(x)

        p1 = self.pool1(x)
        x = torch.cat([x, p1], dim=1)
        x = self.conv3(x)

        p2 = self.pool2(x)
        x = torch.cat([x, p2], dim=1)
        x = self.conv4(x)

        x = self.final(x)
        return x


class EdgesDataset(Dataset):
    def __init__(self, images, image_size=(224, 224), training=True):
        self.images = images
        self.transform = A.Compose([
            A.Compose([
                A.PadIfNeeded(256, 256),
                A.RandomSizedCrop((128, 256), image_size[0], image_size[1]),
                A.RandomRotate90(),
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
                A.Cutout(),
                A.ElasticTransform()
            ], p=float(training)),
            A.Compose([
                A.PadIfNeeded(image_size[0], image_size[1]),
                A.CenterCrop(image_size[0], image_size[1])
            ], p=float(not training))
        ])
        self.normalize = A.Normalize()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)

        data = self.transform(image=image)
        data['mask'] = canny_edges(data['image'])
        data = self.normalize(**data)
        data['image'] = tensor_from_rgb_image(data['image'])
        data['mask'] = torch.from_numpy(data['mask']).float().unsqueeze(0)

        return {'features': data['image'], 'targets': data['mask']}


def visualize_canny_predictions(input, output, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    images = []
    for image, target, logits in zip(input['features'], input['targets'], output['logits']):
        image = rgb_image_from_tensor(image, mean, std)
        target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits.sigmoid()).squeeze(0)

        overlay = np.zeros_like(image)
        overlay[logits > 0.5] += np.array([255, 0, 0], dtype=overlay.dtype)
        overlay[target > 0] += np.array([0, 255, 0], dtype=overlay.dtype)

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        images.append(overlay)
    return images


def main():
    images_dir = 'c:\\datasets\\ILSVRC2013_DET_val'

    canny_cnn = maybe_cuda(CannyModel())
    optimizer = Adam(canny_cnn.parameters(), lr=1e-4)

    images = find_images_in_dir(images_dir)
    train_images, valid_images = train_test_split(images, test_size=0.1, random_state=1234)

    num_workers = 6
    num_epochs = 100
    batch_size = 16

    if False:
        train_images = train_images[:batch_size * 4]
        valid_images = valid_images[:batch_size * 4]

    train_loader = DataLoader(EdgesDataset(train_images), batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              drop_last=True, pin_memory=True)
    valid_loader = DataLoader(EdgesDataset(valid_images), batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    # model runner
    runner = SupervisedRunner()
    # checkpoint = UtilsFactory.load_checkpoint("logs/checkpoints//best.pth")
    # UtilsFactory.unpack_checkpoint(checkpoint, model=canny_cnn)

    # model training
    runner.train(
        model=canny_cnn,
        criterion=FocalLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=[
            JaccardCallback(),
            ShowPolarBatchesCallback(visualize_canny_predictions, metric='jaccard', minimize=False),
            EarlyStoppingCallback(patience=5, min_delta=0.01, metric='jaccard', minimize=False),
        ],
        loaders=loaders,
        logdir='logs',
        num_epochs=num_epochs,
        verbose=True,
        main_metric='jaccard',
        minimize_metric=False
        # check=True
    )


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
