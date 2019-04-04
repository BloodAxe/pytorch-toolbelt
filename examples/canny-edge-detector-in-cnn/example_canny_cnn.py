import collections

import cv2
import numpy as np
import torch
import albumentations as A
import torch.nn.functional as F
from catalyst.contrib.criterion import FocalLoss
from catalyst.dl.callbacks import EarlyStoppingCallback, PrecisionCallback, UtilsFactory, JaccardCallback, Callback, RunnerState, TensorboardLogger
from catalyst.dl.experiments import SupervisedRunner
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pytorch_toolbelt.utils.fs import find_images_in_dir, read_rgb_image
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, tensor_from_rgb_image, rgb_image_from_tensor, to_numpy


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
            A.PadIfNeeded(256, 256),
            A.RandomSizedCrop((128, 256), image_size[0], image_size[1]),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(),
            A.ElasticTransform(),
            A.Cutout()
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


class ShowPolarBatchesCallback(Callback):
    def __init__(
            self,
            visualize_batch,
            metric: str = "loss",
            minimize: bool = True,
            min_delta: float = 1e-6,
    ):
        self.best_score = None
        self.best_input = None
        self.best_output = None

        self.worst_score = None
        self.worst_input = None
        self.worst_output = None

        self.target_metric = metric
        self.num_bad_epochs = 0
        self.is_better = None
        self.visualize_batch = visualize_batch

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
            self.is_worse = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best - min_delta)
            self.is_worse = lambda score, best: score <= (best - min_delta)

    def to_cpu(self, data):
        if isinstance(data, dict):
            return dict((key, self.to_cpu(value)) for (key, value) in data.items())
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        if isinstance(data, list):
            return [self.to_cpu(value) for value in data]

        raise ValueError("Unsupported type", type(data))

    def _log_image(self, loggers, mode: str, image, name, step: int, suffix=""):
        for logger in loggers:
            if isinstance(logger, TensorboardLogger):
                logger.loggers[mode].add_image(f"{name}{suffix}", tensor_from_rgb_image(image), step)

    def on_epoch_start(self, state):
        self.best_score = None
        self.best_input = None
        self.best_output = None

        self.worst_score = None
        self.worst_input = None
        self.worst_output = None

    def on_batch_end(self, state: RunnerState):
        value = state.metrics.batch_values[self.target_metric]

        if self.best_score is None or self.is_better(value, self.best_score):
            self.best_score = value
            self.best_input = self.to_cpu(state.input)
            self.best_output = self.to_cpu(state.output)

        if self.worst_score is None or self.is_worse(value, self.worst_score):
            self.worst_score = value
            self.worst_input = self.to_cpu(state.input)
            self.worst_output = self.to_cpu(state.output)

    def on_epoch_end(self, state: RunnerState) -> None:
        mode = state.loader_name

        if self.best_score is not None:
            best_samples = self.visualize_batch(self.best_input, self.best_output)
            for i, image in enumerate(best_samples):
                self._log_image(state.loggers, mode, image, f"Best Batch/{i}", state.step, suffix="/epoch")
                cv2.imshow("Best sample " + str(i), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if self.worst_score is not None:
            worst_samples = self.visualize_batch(self.worst_input, self.worst_output)
            for i, image in enumerate(worst_samples):
                self._log_image(state.loggers, mode, image, f"Worst Batch/{i}", state.step, suffix="/epoch")
                cv2.imshow("Worst sample " + str(i), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        cv2.waitKey(1000)


def visualize_canny_predictions(input, output, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    images = []
    for image, target, logits in zip(input['features'], input['targets'], output['logits']):
        image = rgb_image_from_tensor(image, mean, std)
        target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits.sigmoid()).squeeze(0)

        overlay = np.ones_like(image) * (255, 0, 0)  # Full red image
        overlay = overlay * np.expand_dims(logits, -1)
        overlay[target > 0] = (0, 255, 0)

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        images.append(overlay)
    return images


def main():
    images_dir = 'd:\datasets\mirflickr'

    canny_cnn = maybe_cuda(CannyModel())
    optimizer = Adam(canny_cnn.parameters(), lr=1e-4)

    images = find_images_in_dir(images_dir)
    train_images, valid_images = train_test_split(images, test_size=0.1, random_state=1234)

    num_workers = 0
    num_epochs = 100
    batch_size = 128

    if True:
        batch_size = 32
        train_images = train_images[:batch_size * 4]
        valid_images = valid_images[:batch_size * 4]

    train_loader = DataLoader(EdgesDataset(train_images), batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(EdgesDataset(valid_images), batch_size=batch_size, num_workers=num_workers, pin_memory=True)
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

    UtilsFactory.plot_metrics(
        logdir='logs',
        metrics=["loss", "jaccard", "base/lr"])


if __name__ == '__main__':
    main()
