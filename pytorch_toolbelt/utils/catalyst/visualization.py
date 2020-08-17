import warnings
from typing import Callable, Optional, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from catalyst.dl import Callback, CallbackOrder, IRunner
from catalyst.dl.callbacks import TensorboardLogger
from catalyst.contrib.tools.tensorboard import SummaryWriter

from ..torch_utils import rgb_image_from_tensor, to_numpy
from ..torch_utils import tensor_from_rgb_image

__all__ = [
    "get_tensorboard_logger",
    "ShowPolarBatchesCallback",
    "ShowEmbeddingsCallback",
    "draw_binary_segmentation_predictions",
    "draw_semantic_segmentation_predictions",
]


def get_tensorboard_logger(runner: IRunner, tensorboard_callback_name: str = "_tensorboard") -> SummaryWriter:
    tb_callback: TensorboardLogger = runner.callbacks[tensorboard_callback_name]
    if runner.loader_name not in tb_callback.loggers:
        raise RuntimeError(f"Cannot find Tensorboard logger for loader {runner.loader_name}")
    return tb_callback.loggers[runner.loader_name]


class ShowPolarBatchesCallback(Callback):
    """
    Visualize best and worst batch based in metric in Tensorboard
    """

    def __init__(
        self,
        visualize_batch: Callable,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6,
        targets="tensorboard",
    ):
        """

        :param visualize_batch: Visualization function that must return list of images.
               It's takes two arguments: (batch input, predicted output).
        :param metric:
        :param minimize:
        :param min_delta:
        :param targets: Str 'tensorboard' or 'matplotlib, or ['tensorboard', 'matplotlib']
        """
        super().__init__(CallbackOrder.Other)
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
        if isinstance(data, (list, tuple)):
            return [self.to_cpu(value) for value in data]
        return data

    def on_loader_start(self, runner):
        self.best_score = None
        self.best_input = None
        self.best_output = None

        self.worst_score = None
        self.worst_input = None
        self.worst_output = None

    def on_batch_end(self, runner: IRunner):
        value = runner.batch_metrics.get(self.target_metric, None)
        if value is None:
            warnings.warn(f"Metric value for {self.target_metric} is not available in runner.metrics.batch_values")
            return

        if self.best_score is None or self.is_better(value, self.best_score):
            self.best_score = value
            self.best_input = self.to_cpu(runner.input)
            self.best_output = self.to_cpu(runner.output)

        if self.worst_score is None or self.is_worse(value, self.worst_score):
            self.worst_score = value
            self.worst_input = self.to_cpu(runner.input)
            self.worst_output = self.to_cpu(runner.output)

    def on_loader_end(self, runner: IRunner):
        logger = get_tensorboard_logger(runner)

        if self.best_score is not None:
            best_samples = self.visualize_batch(self.best_input, self.best_output)
            self._log_samples(best_samples, "best", logger, runner.global_batch_step)

        if self.worst_score is not None:
            worst_samples = self.visualize_batch(self.worst_input, self.worst_output)
            self._log_samples(worst_samples, "worst", logger, runner.global_batch_step)

    def _log_samples(self, samples, name, logger, step):
        if "tensorboard" in self.targets:
            for i, image in enumerate(samples):
                logger.add_image(f"{self.target_metric}/{name}/{i}", tensor_from_rgb_image(image), step)

        if "matplotlib" in self.targets:
            for i, image in enumerate(samples):
                plt.figure()
                plt.imshow(image)
                plt.tight_layout()
                plt.axis("off")
                plt.show()


class ShowEmbeddingsCallback(Callback):
    def __init__(
        self,
        embedding_key,
        input_key,
        targets_key,
        prefix="embedding",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__(CallbackOrder.Other)
        self.prefix = prefix
        self.embedding_key = embedding_key
        self.input_key = input_key
        self.targets_key = targets_key
        self.mean = torch.tensor(mean).view((1, 3, 1, 1))
        self.std = torch.tensor(std).view((1, 3, 1, 1))

        self.embeddings = []
        self.images = []
        self.targets = []

    def on_loader_start(self, runner: IRunner):
        self.embeddings = []
        self.images = []
        self.targets = []

    def on_loader_end(self, runner: IRunner):
        logger = get_tensorboard_logger(runner)
        logger.add_embedding(
            mat=torch.cat(self.embeddings, dim=0),
            metadata=self.targets,
            label_img=torch.cat(self.images, dim=0),
            global_step=runner.epoch,
            tag=self.prefix,
        )

    def on_batch_end(self, runner: IRunner):
        embedding = runner.output[self.embedding_key].detach().cpu()
        image = runner.input[self.input_key].detach().cpu()
        targets = runner.input[self.targets_key].detach().cpu().tolist()

        image = F.interpolate(image, size=(256, 256))
        image = image * self.std + self.mean

        self.images.append(image)
        self.embeddings.append(embedding)
        self.targets.extend(targets)


def draw_binary_segmentation_predictions(
    input: dict,
    output: dict,
    image_key="features",
    image_id_key: Optional[str] = "image_id",
    targets_key="targets",
    outputs_key="logits",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_images=None,
    targets_threshold=0.5,
    logits_threshold=0,
    image_format: Union[str, Callable] = "rgb",
) -> List[np.ndarray]:
    """
    Render visualization of model's prediction for binary segmentation problem.
    This function draws a color-coded overlay on top of the image, with color codes meaning:
        - green: True positives
        - red: False-negatives
        - yellow: False-positives

    :param input: Input batch (model's input batch)
    :param output: Output batch (model predictions)
    :param image_key: Key for getting image
    :param image_id_key: Key for getting image id/fname
    :param targets_key: Key for getting ground-truth mask
    :param outputs_key: Key for getting model logits for predicted mask
    :param mean: Mean vector user during normalization
    :param std: Std vector user during normalization
    :param max_images: Maximum number of images to visualize from batch
        (If you have huge batch, saving hundreds of images may make TensorBoard slow)
    :param targets_threshold: Threshold to convert target values to binary.
        Default value 0.5 is safe for both smoothed and hard labels.
    :param logits_threshold: Threshold to convert model predictions (raw logits) values to binary.
        Default value 0.0 is equivalent to 0.5 after applying sigmoid activation
    :param image_format: Source format of the image tensor to conver to RGB representation.
        Can be string ("gray", "rgb", "brg") or function `convert(np.ndarray)->nd.ndarray`.
    :return: List of images
    """
    images = []
    num_samples = len(input[image_key])
    if max_images is not None:
        num_samples = min(num_samples, max_images)

    assert output[outputs_key].size(1) == 1, "Mask must be single-channel tensor of shape [Nx1xHxW]"

    for i in range(num_samples):
        image = rgb_image_from_tensor(input[image_key][i], mean, std)

        if image_format == "rgb":
            pass
        elif image_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image_format, callable):
            image = image_format(image)

        target = to_numpy(input[targets_key][i]).squeeze(0)
        logits = to_numpy(output[outputs_key][i]).squeeze(0)

        overlay = image.copy()
        true_mask = target > targets_threshold
        pred_mask = logits > logits_threshold

        overlay[true_mask & pred_mask] = np.array(
            [0, 250, 0], dtype=overlay.dtype
        )  # Correct predictions (Hits) painted with green
        overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        overlay[~true_mask & pred_mask] = np.array(
            [250, 250, 0], dtype=overlay.dtype
        )  # False alarm painted with yellow
        overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)

        if image_id_key is not None and image_id_key in input:
            image_id = input[image_id_key][i]
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)
    return images


def draw_semantic_segmentation_predictions(
    input: dict,
    output: dict,
    class_colors: List,
    mode="overlay",
    image_key="features",
    image_id_key="image_id",
    targets_key="targets",
    outputs_key="logits",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_images=None,
    image_format: Union[str, Callable] = "rgb",
) -> List[np.ndarray]:
    """
    Render visualization of model's prediction for binary segmentation problem.
    This function draws a color-coded overlay on top of the image, with color codes meaning:
        - green: True positives
        - red: False-negatives
        - yellow: False-positives

    :param input: Input batch (model's input batch)
    :param output: Output batch (model predictions)
    :param class_colors:
    :param mode:
    :param image_key: Key for getting image
    :param image_id_key: Key for getting image id/fname
    :param targets_key: Key for getting ground-truth mask
    :param outputs_key: Key for getting model logits for predicted mask
    :param mean: Mean vector user during normalization
    :param std: Std vector user during normalization
    :param max_images: Maximum number of images to visualize from batch
        (If you have huge batch, saving hundreds of images may make TensorBoard slow)
    :param image_format: Source format of the image tensor to conver to RGB representation.
        Can be string ("gray", "rgb", "brg") or function `convert(np.ndarray)->nd.ndarray`.
    :return: List of images
    """
    assert mode in {"overlay", "side-by-side"}

    images = []
    num_samples = len(input[image_key])
    if max_images is not None:
        num_samples = min(num_samples, max_images)

    for i in range(num_samples):
        image = rgb_image_from_tensor(input[image_key][i], mean, std)

        if image_format == "rgb":
            pass
        elif image_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image_format, callable):
            image = image_format(image)

        target = to_numpy(input[targets_key][i])
        logits = to_numpy(output[outputs_key][i]).argmax(axis=0)

        if mode == "overlay":
            overlay = image.copy()
            for class_index, class_color in enumerate(class_colors):
                overlay[logits == class_index, :] = class_color

            overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        elif mode == "side-by-side":
            true_mask = np.zeros_like(image)
            for class_index, class_color in enumerate(class_colors):
                true_mask[target == class_index, :] = class_color

            pred_mask = np.zeros_like(image)
            for class_index, class_color in enumerate(class_colors):
                pred_mask[logits == class_index, :] = class_color

            overlay = np.hstack((image, true_mask, pred_mask))
        else:
            raise ValueError(mode)

        if image_id_key is not None and image_id_key in input:
            image_id = input[image_id_key][i]
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)

    return images


def draw_multilabel_segmentation_predictions(
    input: dict,
    output: dict,
    class_colors: List,
    mode="side-by-side",
    image_key="features",
    image_id_key="image_id",
    targets_key="targets",
    outputs_key="logits",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_images=None,
    targets_threshold=0.5,
    logits_threshold=0,
    image_format: Union[str, Callable] = "rgb",
) -> List[np.ndarray]:
    """
    Render visualization of model's prediction for binary segmentation problem.
    This function draws a color-coded overlay on top of the image, with color codes meaning:
        - green: True positives
        - red: False-negatives
        - yellow: False-positives

    :param input: Input batch (model's input batch)
    :param output: Output batch (model predictions)
    :param class_colors:
    :param mode:
    :param image_key: Key for getting image
    :param image_id_key: Key for getting image id/fname
    :param targets_key: Key for getting ground-truth mask
    :param outputs_key: Key for getting model logits for predicted mask
    :param mean: Mean vector user during normalization
    :param std: Std vector user during normalization
    :param max_images: Maximum number of images to visualize from batch
        (If you have huge batch, saving hundreds of images may make TensorBoard slow)
    :param image_format: Source format of the image tensor to conver to RGB representation.
        Can be string ("gray", "rgb", "brg") or function `convert(np.ndarray)->nd.ndarray`.
    :return: List of images
    """
    assert mode in {"overlay", "side-by-side"}

    images = []
    num_samples = len(input[image_key])
    if max_images is not None:
        num_samples = min(num_samples, max_images)

    for i in range(num_samples):
        image = rgb_image_from_tensor(input[image_key][i], mean, std)

        if image_format == "rgb":
            pass
        elif image_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_format == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif isinstance(image_format, callable):
            image = image_format(image)

        target = to_numpy(input[targets_key][i]) > targets_threshold
        logits = to_numpy(output[outputs_key][i]) > logits_threshold

        if mode == "overlay":
            overlay = image.copy()
            for class_index, class_color in enumerate(class_colors):
                overlay[logits[class_index], :] = class_color

            overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        elif mode == "side-by-side":
            true_mask = image.copy()
            for class_index, class_color in enumerate(class_colors):
                true_mask[target[class_index], :] = class_color

            pred_mask = image.copy()
            for class_index, class_color in enumerate(class_colors):
                pred_mask[logits[class_index], :] = class_color

            true_mask = cv2.addWeighted(image, 0.5, true_mask, 0.5, 0, dtype=cv2.CV_8U)
            pred_mask = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0, dtype=cv2.CV_8U)
            overlay = np.hstack((true_mask, pred_mask))
        else:
            raise ValueError(mode)

        if image_id_key is not None and image_id_key in input:
            image_id = input[image_id_key][i]
            cv2.putText(overlay, str(image_id), (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 250))

        images.append(overlay)

    return images
