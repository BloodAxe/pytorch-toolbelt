import numpy as np
import torch
from catalyst.dl import Callback, RunnerState, MetricCallback
from catalyst.dl.callbacks import TensorboardLogger
from sklearn.metrics import f1_score, confusion_matrix
from tensorboardX import SummaryWriter

from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import render_figure_to_tensor, plot_confusion_matrix

__all__ = ['pixel_accuracy',
           'binary_iou_score',
           'multiclass_iou_score',
           'multilabel_iou_score',
           'PixelAccuracyCallback',
           'MacroF1Callback',
           'ConfusionMatrixCallback',
           'JaccardScoreCallback']


def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
    for logger in state.loggers:
        if isinstance(logger, TensorboardLogger):
            return logger.loggers[state.loader_name]
    raise RuntimeError(f"Cannot find Tensorboard logger for loader {state.loader_name}")


def pixel_accuracy(outputs, targets):
    """Compute the pixel accuracy
    """
    outputs = (outputs.detach() > 0).float()

    correct = float(torch.sum(outputs == targets.detach()))
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
    ):
        """
        :param input_key: input key to use for iou calculation;
            specifies our `y_true`.
        :param output_key: output key to use for iou calculation;
            specifies our `y_pred`
        """
        super().__init__(
            prefix=prefix,
            metric_fn=pixel_accuracy,
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
            class_names=None
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        self.prefix = prefix
        self.class_names = class_names
        self.output_key = output_key
        self.input_key = input_key
        self.outputs = []
        self.targets = []

    def on_loader_start(self, state):
        self.outputs = []
        self.targets = []

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(state.output[self.output_key])
        targets = to_numpy(state.input[self.input_key])

        outputs = np.argmax(outputs, axis=1)

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
        cm = confusion_matrix(outputs, targets, labels=range(num_classes))

        fig = plot_confusion_matrix(cm,
                                    figsize=(6 + num_classes // 3, 6 + num_classes // 3),
                                    class_names=class_names,
                                    normalize=True,
                                    noshow=True)
        fig = render_figure_to_tensor(fig)

        logger = _get_tensorboard_logger(state)
        logger.add_image(f'{self.prefix}/epoch', fig, global_step=state.step)


class MacroF1Callback(Callback):
    """
    Compute F1-macro metric
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "macro_f1"
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

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(state.output[self.output_key])
        targets = to_numpy(state.input[self.input_key])
        num_classes = outputs.shape[1]

        outputs = [np.eye(num_classes)[y] for y in np.argmax(outputs, axis=1)]
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


def multiclass_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=1)

    for class_index in range(num_classes):
        iou = binary_iou_score((y_true == class_index).float(),
                               (y_pred == class_index).float(), threshold, eps)
        ious.append(iou)

    return ious


def multilabel_iou_score(y_true: torch.Tensor, y_pred: torch.Tensor, threshold=0., eps=1e-3):
    ious = []
    num_classes = y_pred.size(0)

    for class_index in range(num_classes):
        iou = binary_iou_score(y_true[class_index], y_pred[class_index], threshold, eps)
        ious.append(iou)

    return ious


class JaccardScoreCallback(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self,
                 mode: str,
                 class_names=None,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "jaccard"):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        assert mode in {'binary', 'multiclass', 'multilabel'}

        self.mode = mode
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.class_names = class_names
        self.scores = []

    def on_loader_start(self, state):
        self.scores = []

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        score_fn = None

        if self.mode == 'binary':
            assert outputs.size(1) == 1
            assert targets.size(1) == 1
            score_fn = binary_iou_score

        if self.mode == 'multiclass':
            assert targets.size(1) == 1
            score_fn = multiclass_iou_score

        if self.mode == 'multilabel':
            assert outputs.size(1) == targets.size(1)
            score_fn = multilabel_iou_score

        assert score_fn is not None

        batch_size = targets.size(0)
        for image_index in range(batch_size):
            self.scores.append(score_fn(targets[image_index], outputs[image_index]))

    def on_loader_end(self, state):
        scores = np.array(self.scores)

        state.metrics.epoch_values[state.loader_name][self.prefix] = float(np.mean(scores))

        # Log additional IoU scores per class
        if self.mode in {'multiclass', 'multilabel'}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f'class_{i}' for i in range(num_classes)]

            scores_per_class = np.mean(scores, dim=1)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                state.metrics.epoch_values[state.loader_name][self.prefix + '_' + class_name] = float(score_per_class)
