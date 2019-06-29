import warnings
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from catalyst.dl import Callback, RunnerState
from catalyst.dl.callbacks import TensorboardLogger
from tensorboardX import SummaryWriter

from pytorch_toolbelt.utils.torch_utils import rgb_image_from_tensor, to_numpy
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

__all__ = [
    'get_tensorboard_logger',
    'ShowPolarBatchesCallback',
    'draw_binary_segmentation_predictions',
    'draw_semantic_segmentation_predictions']


def get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
    for logger in state.loggers:
        if isinstance(logger, TensorboardLogger):
            return logger.loggers[state.loader_name]
    raise RuntimeError(f"Cannot find Tensorboard logger for loader {state.loader_name}")


class ShowPolarBatchesCallback(Callback):
    """
    Visualize best and worst batch based in metric in Tensorboard
    """

    def __init__(self,
                 visualize_batch: Callable,
                 metric: str = "loss",
                 minimize: bool = True,
                 min_delta: float = 1e-6,
                 targets='tensorboard'):
        """

        :param visualize_batch: Visualization function that must return list of images.
               It's takes two arguments: (batch input, predicted output).
        :param metric:
        :param minimize:
        :param min_delta:
        :param targets: Str 'tensorboard' or 'matplotlib, or ['tensorboard', 'matplotlib']
        """
        assert isinstance(targets, (list, str))

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
        self.targets = [targets] if isinstance(targets, str) else targets

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
            self.is_worse = lambda score, worst: score >= (worst + min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)
            self.is_worse = lambda score, worst: score <= (worst - min_delta)

    def to_cpu(self, data):
        if isinstance(data, dict):
            return dict((key, self.to_cpu(value)) for (key, value) in data.items())
        if isinstance(data, torch.Tensor):
            return data.detach().cpu()
        if isinstance(data, list):
            return [self.to_cpu(value) for value in data]
        if isinstance(data, str):
            return data
        raise ValueError("Unsupported type", type(data))

    def on_loader_start(self, state):
        self.best_score = None
        self.best_input = None
        self.best_output = None

        self.worst_score = None
        self.worst_input = None
        self.worst_output = None

    def on_batch_end(self, state: RunnerState):
        value = state.metrics.batch_values.get(self.target_metric, None)
        if value is None:
            warnings.warn(f'Metric value for {self.target_metric} is not available in state.metrics.batch_values')
            return

        if self.best_score is None or self.is_better(value, self.best_score):
            self.best_score = value
            self.best_input = self.to_cpu(state.input)
            self.best_output = self.to_cpu(state.output)

        if self.worst_score is None or self.is_worse(value, self.worst_score):
            self.worst_score = value
            self.worst_input = self.to_cpu(state.input)
            self.worst_output = self.to_cpu(state.output)

    def on_loader_end(self, state: RunnerState) -> None:
        logger = _get_tensorboard_logger(state)

        if self.best_score is not None:
            best_samples = self.visualize_batch(self.best_input, self.best_output)

            if 'tensorboard' in self.targets:
                for i, image in enumerate(best_samples):
                    logger.add_image(f"{self.target_metric}/best/{i}", tensor_from_rgb_image(image), state.step)

            if 'matplotlib' in self.targets:
                for i, image in enumerate(best_samples):
                    plt.figure()
                    plt.imshow(image)
                    plt.tight_layout()
                    plt.axis('off')
                    plt.show()

        if self.worst_score is not None:
            worst_samples = self.visualize_batch(self.worst_input, self.worst_output)
            if 'tensorboard' in self.targets:
                for i, image in enumerate(worst_samples):
                    logger.add_image(f"{self.target_metric}/worst/{i}", tensor_from_rgb_image(image), state.step)

            if 'matplotlib' in self.targets:
                for i, image in enumerate(worst_samples):
                    plt.figure()
                    plt.imshow(image)
                    plt.tight_layout()
                    plt.axis('off')
                    plt.show()


def draw_binary_segmentation_predictions(input: dict,
                                         output: dict,
                                         image_key='features',
                                         image_id_key='image_id',
                                         targets_key='targets',
                                         outputs_key='logits',
                                         mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)):
    images = []
    image_id_input = input[image_id_key] if image_id_key is not None else [None] * len(input[image_key])

    for image, target, image_id, logits in zip(input[image_key],
                                               input[targets_key],
                                               image_id_input,
                                               output[outputs_key]):
        image = rgb_image_from_tensor(image, mean, std)
        target = to_numpy(target).squeeze(0)
        logits = to_numpy(logits).squeeze(0)

        overlay = image.copy()
        true_mask = target > 0
        pred_mask = logits > 0

        overlay[true_mask & pred_mask] = np.array([0, 250, 0], dtype=overlay.dtype)  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array([250, 250, 0], dtype=overlay.dtype)  # False alarm painted with yellow
        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)

        if image_id is not None:
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images


def draw_semantic_segmentation_predictions(input: dict,
                                           output: dict,
                                           class_colors,
                                           image_key='features',
                                           image_id_key='image_id',
                                           targets_key='targets',
                                           outputs_key='logits',
                                           mean=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225)):
    images = []
    image_id_input = input[image_id_key] if image_id_key is not None else [None] * len(input[image_key])

    for image, target, image_id, logits in zip(input[image_key],
                                               input[targets_key],
                                               image_id_input,
                                               output[outputs_key]):
        image = rgb_image_from_tensor(image, mean, std)

        logits = to_numpy(logits).argmax(axis=0)

        overlay = image.copy()
        for class_index, class_color in enumerate(range(len(class_colors))):
            image[logits == class_index, :] = class_color

        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)

        if image_id is not None:
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)

    return images
