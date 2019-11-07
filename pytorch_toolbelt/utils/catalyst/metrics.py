from functools import partial

import numpy as np
import torch
from catalyst.dl import Callback, RunnerState, MetricCallback, CallbackOrder
from pytorch_toolbelt.utils.catalyst.visualization import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import (
    render_figure_to_tensor,
    plot_confusion_matrix,
)
from sklearn.metrics import f1_score, confusion_matrix

__all__ = [
    "pixel_accuracy",
    "binary_dice_iou_score",
    "multiclass_dice_iou_score",
    "multilabel_dice_iou_score",
    "PixelAccuracyCallback",
    "MacroF1Callback",
    "ConfusionMatrixCallback",
    "IoUMetricsCallback",
]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def pixel_accuracy(outputs: torch.Tensor, targets: torch.Tensor, ignore_index=None):
    """
    Compute the pixel accuracy
    """
    outputs = outputs.detach()
    targets = targets.detach()
    if ignore_index is not None:
        mask = targets != ignore_index
        outputs = outputs[mask]
        targets = targets[mask]

    outputs = (outputs > 0).float()

    correct = float(torch.sum(outputs == targets))
    total = targets.numel()
    return correct / total


class PixelAccuracyCallback(MetricCallback):
    """Pixel accuracy metric callback
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        ignore_index=None,
    ):
        """
        :param input_key: input key to use for iou calculation;
            specifies our `y_true`.
        :param output_key: output key to use for iou calculation;
            specifies our `y_pred`
        :param ignore_index: same meaning as in nn.CrossEntropyLoss
        """
        super().__init__(
            prefix=prefix,
            metric_fn=partial(pixel_accuracy, ignore_index=ignore_index),
            input_key=input_key,
            output_key=output_key,
        )


class ConfusionMatrixCallback(Callback):
    """
    Compute and log confusion matrix to Tensorboard.
    For use with Multiclass classification/segmentation.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        class_names=None,
        ignore_index=None,
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param ignore_index: same meaning as in nn.CrossEntropyLoss
        """
        super().__init__(CallbackOrder.Logger)
        self.prefix = prefix
        self.class_names = class_names
        self.output_key = output_key
        self.input_key = input_key
        self.outputs = []
        self.targets = []
        self.ignore_index = ignore_index

    def on_loader_start(self, state):
        self.outputs = []
        self.targets = []

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(state.output[self.output_key])
        targets = to_numpy(state.input[self.input_key])

        outputs = np.argmax(outputs, axis=1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            outputs = outputs[mask]
            targets = targets[mask]

        self.outputs.extend(outputs)
        self.targets.extend(targets)

    def on_loader_end(self, state):
        targets = np.array(self.targets)
        outputs = np.array(self.outputs)

        if self.class_names is None:
            class_names = [str(i) for i in range(targets.shape[1])]
        else:
            class_names = self.class_names

        num_classes = len(class_names)
        cm = confusion_matrix(targets, outputs, labels=range(num_classes))

        fig = plot_confusion_matrix(
            cm,
            figsize=(6 + num_classes // 3, 6 + num_classes // 3),
            class_names=class_names,
            normalize=True,
            noshow=True,
        )
        fig = render_figure_to_tensor(fig)

        logger = get_tensorboard_logger(state)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=state.step)


class MacroF1Callback(Callback):
    """
    Compute F1-macro metric
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "macro_f1",
        ignore_index=None,
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.metric_fn = lambda outputs, targets: f1_score(
            targets, outputs, average="macro"
        )
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.outputs = []
        self.targets = []
        self.ignore_index = ignore_index

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(state.output[self.output_key])
        targets = to_numpy(state.input[self.input_key])

        num_classes = outputs.shape[1]
        outputs = np.argmax(outputs, axis=1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            outputs = outputs[mask]
            targets = targets[mask]

        outputs = [np.eye(num_classes)[y] for y in outputs]
        targets = [np.eye(num_classes)[y] for y in targets]

        self.outputs.extend(outputs)
        self.targets.extend(targets)

        # metric = self.metric_fn(self.targets, self.outputs)
        # state.metrics.add_batch_value(name=self.prefix, value=metric)

    def on_loader_start(self, state):
        self.outputs = []
        self.targets = []

    def on_loader_end(self, state):
        metric_name = self.prefix
        targets = np.array(self.targets)
        outputs = np.array(self.outputs)

        metric = self.metric_fn(outputs, targets)
        state.metrics.epoch_values[state.loader_name][metric_name] = metric


def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    nan_score_on_empty=False,
    eps=1e-7,
) -> float:
    """
    Compute IoU score between two image tensors
    :param y_pred: Input image tensor of any shape
    :param y_true: Target image of any shape (must match size of y_pred)
    :param mode: Metric to compute (dice, iou)
    :param threshold: Optional binarization threshold to apply on @y_pred
    :param nan_score_on_empty: If true, return np.nan if target has no positive pixels;
        If false, return 1. if both target and input are empty, and 0 otherwise.
    :param eps: Small value to add to denominator for numerical stability
    :return: Float scalar
    """
    assert mode in {"dice", "iou"}

    # Binarize predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score


def multiclass_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_dice_iou_score(
            y_pred=(y_pred == class_index).float(),
            y_true=(y_true == class_index).float(),
            mode=mode,
            nan_score_on_empty=nan_score_on_empty,
            threshold=threshold,
            eps=eps,
        )
        ious.append(iou)

    return ious


def multilabel_dice_iou_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
):
    ious = []
    num_classes = y_pred.size(0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_dice_iou_score(
            y_pred=y_pred[class_index],
            y_true=y_true[class_index],
            mode=mode,
            threshold=threshold,
            nan_score_on_empty=nan_score_on_empty,
            eps=eps,
        )
        ious.append(iou)

    return ious


class IoUMetricsCallback(Callback):
    """
    A metric callback for computing either Dice or Jaccard metric
    which is computed across whole epoch, not per-batch.
    """

    def __init__(
        self,
        mode: str,
        metric="dice",
        class_names=None,
        classes_of_interest=None,
        input_key: str = "targets",
        output_key: str = "logits",
        nan_score_on_empty=True,
        prefix: str = None,
        threshold: float = 0.0
    ):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        :param threshold: Optional binarization threshold to apply on `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}

        if prefix is None:
            prefix = metric

        if classes_of_interest is not None:
            if classes_of_interest.dtype == np.bool:
                num_classes = len(classes_of_interest)
                classes_of_interest = np.arange(num_classes)[classes_of_interest]

            if class_names is not None:
                if len(class_names) != len(classes_of_interest):
                    raise ValueError(
                        "Length of 'classes_of_interest' must be equal to length of 'classes_of_interest'"
                    )

        self.mode = mode
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.classes_of_interest = classes_of_interest
        self.scores = []

        if self.mode == BINARY_MODE:
            self.score_fn = partial(
                binary_dice_iou_score,
                threshold=threshold,
                nan_score_on_empty=nan_score_on_empty,
                mode=metric,
            )

        if self.mode == MULTICLASS_MODE:
            self.score_fn = partial(
                multiclass_dice_iou_score,
                mode=metric,
                threshold=threshold,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
            )

        if self.mode == MULTILABEL_MODE:
            self.score_fn = partial(
                multilabel_dice_iou_score,
                mode=metric,
                threshold=threshold,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
            )

    def on_loader_start(self, state):
        self.scores = []

    @torch.no_grad()
    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        batch_size = targets.size(0)
        score_per_image = []
        for image_index in range(batch_size):
            score_per_class = self.score_fn(
                y_pred=outputs[image_index], y_true=targets[image_index]
            )
            score_per_image.append(score_per_class)

        mean_score = np.nanmean(score_per_image)
        state.metrics.add_batch_value(self.prefix, float(mean_score))
        self.scores.extend(score_per_image)

    def on_loader_end(self, state):
        scores = np.array(self.scores)
        mean_score = np.nanmean(scores)

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(mean_score)

        # Log additional IoU scores per class
        if self.mode in {MULTICLASS_MODE, MULTILABEL_MODE}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f"class_{i}" for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                state.metrics.epoch_values[state.loader_name][
                    self.prefix + "_" + class_name
                ] = float(score_per_class)
