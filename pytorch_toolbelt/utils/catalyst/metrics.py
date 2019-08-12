from functools import partial

import numpy as np
import torch
from catalyst.dl import Callback, RunnerState, MetricCallback
from sklearn.metrics import f1_score, confusion_matrix

from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import render_figure_to_tensor, plot_confusion_matrix
from pytorch_toolbelt.utils.catalyst.visualization import get_tensorboard_logger

__all__ = ['pixel_accuracy',
           'binary_iou_score',
           'multiclass_iou_score',
           'multilabel_iou_score',
           'PixelAccuracyCallback',
           'MacroF1Callback',
           'ConfusionMatrixCallback',
           'JaccardScoreCallback']


def pixel_accuracy(outputs: torch.Tensor,
                   targets: torch.Tensor, ignore_index=None):
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
            ignore_index=None
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
            ignore_index=None
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param ignore_index: same meaning as in nn.CrossEntropyLoss
        """
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

        fig = plot_confusion_matrix(cm,
                                    figsize=(6 + num_classes // 3, 6 + num_classes // 3),
                                    class_names=class_names,
                                    normalize=True,
                                    noshow=True)
        fig = render_figure_to_tensor(fig)

        logger = get_tensorboard_logger(state)
        logger.add_image(f'{self.prefix}/epoch', fig, global_step=state.step)


class MacroF1Callback(Callback):
    """
    Compute F1-macro metric
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "macro_f1",
            ignore_index=None
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        self.metric_fn = lambda outputs, targets: f1_score(targets, outputs, average='macro')
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


def binary_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3):
    if y_true.sum() == 0:
        return np.nan

    # Binarize predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    iou = intersection / (union - intersection + eps)
    return float(iou)


def multiclass_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3, classes_of_interest=None):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_iou_score((y_true == class_index).float(),
                               (y_pred == class_index).float(), threshold, eps)
        ious.append(iou)

    return ious


def multilabel_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3, classes_of_interest=None):
    ious = []
    num_classes = y_pred.size(0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        iou = binary_iou_score(y_true[class_index], y_pred[class_index], threshold, eps)
        ious.append(iou)

    return ious


class JaccardScoreCallback(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self,
                 mode: str,
                 num_classes: int = None,
                 class_names=None,
                 classes_of_interest=None,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "jaccard"):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        assert mode in {'binary', 'multiclass', 'multilabel'}

        if classes_of_interest is not None:
            if classes_of_interest.dtype == np.bool:
                num_classes = len(classes_of_interest)
                classes_of_interest = np.arange(num_classes)[classes_of_interest]

            if class_names is not None:
                if len(class_names) != len(classes_of_interest):
                    raise ValueError('Length of \'classes_of_interest\' must be equal to length of \'classes_of_interest\'')

        self.mode = mode
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.classes_of_interest = classes_of_interest
        self.scores = []

        if self.mode == 'binary':
            self.score_fn = binary_iou_score

        if self.mode == 'multiclass':
            self.score_fn = partial(multiclass_iou_score, classes_of_interest=self.classes_of_interest)

        if self.mode == 'multilabel':
            self.score_fn = partial(multilabel_iou_score, classes_of_interest=self.classes_of_interest)

    def on_loader_start(self, state):
        self.scores = []

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        batch_size = targets.size(0)
        ious = []
        for image_index in range(batch_size):
            iou_per_class = self.score_fn(targets[image_index], outputs[image_index])
            ious.append(iou_per_class)

        iou_per_batch = np.nanmean(ious)
        state.metrics.add_batch_value(self.prefix, float(iou_per_batch))
        self.scores.extend(ious)

    def on_loader_end(self, state):
        scores = np.array(self.scores)
        iou = np.nanmean(scores)

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(iou)

        # Log additional IoU scores per class
        if self.mode in {'multiclass', 'multilabel'}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f'class_{i}' for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                state.metrics.epoch_values[state.loader_name][self.prefix + '_' + class_name] = float(score_per_class)
