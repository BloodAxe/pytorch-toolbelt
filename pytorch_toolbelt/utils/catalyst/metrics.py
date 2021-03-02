import warnings
from functools import partial
from typing import List, Optional, Callable

import numpy as np
import torch
from catalyst.dl import Callback, MetricCallback, CallbackOrder, IRunner
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from torch import Tensor

from .visualization import get_tensorboard_logger
from ..distributed import all_gather, is_main_process
from ..torch_utils import to_numpy, argmax_over_dim_1
from ..visualization import render_figure_to_tensor, plot_confusion_matrix

__all__ = [
    "AccuracyCallback",
    "MultilabelAccuracyCallback",
    "BINARY_MODE",
    "ConfusionMatrixCallback",
    "F1ScoreCallback",
    "IoUMetricsCallback",
    "MULTICLASS_MODE",
    "MULTILABEL_MODE",
    "OutputDistributionCallback",
    "PixelAccuracyCallback",
    "binary_dice_iou_score",
    "multiclass_dice_iou_score",
    "multilabel_dice_iou_score",
    "pixel_accuracy",
]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


@torch.no_grad()
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
    """Pixel accuracy metric callback"""

    def __init__(
        self, input_key: str = "targets", output_key: str = "logits", prefix: str = "accuracy", ignore_index=None
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
        outputs_to_labels: Callable[[Tensor], Tensor] = argmax_over_dim_1,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        class_names: List[str] = None,
        num_classes: int = None,
        ignore_index=None,
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param ignore_index: same meaning as in nn.CrossEntropyLoss
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.class_names = class_names
        self.num_classes = num_classes if class_names is None else len(class_names)
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.confusion_matrix = None
        self.outputs_to_labels = outputs_to_labels

    def on_loader_start(self, state):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.long)

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        pred_labels = self.outputs_to_labels(runner.output[self.output_key])
        true_labels = runner.input[self.input_key].type_as(pred_labels)

        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        if self.ignore_index is not None:
            mask = true_labels != self.ignore_index
            pred_labels = torch.masked_select(pred_labels, mask)
            true_labels = torch.masked_select(true_labels, mask)

        if len(true_labels):
            true_labels = to_numpy(true_labels)
            pred_labels = to_numpy(pred_labels)
            batch_cm = confusion_matrix(
                y_true=true_labels, y_pred=pred_labels, labels=np.arange(self.num_classes, dtype=int)
            )
            self.confusion_matrix = self.confusion_matrix + batch_cm

    def on_loader_end(self, runner: IRunner):
        if self.class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
        else:
            class_names = self.class_names

        num_classes = len(class_names)
        cm = np.sum(all_gather(self.confusion_matrix), axis=0)

        fig = plot_confusion_matrix(
            cm,
            figsize=(6 + num_classes // 3, 6 + num_classes // 3),
            class_names=class_names,
            normalize=True,
            noshow=True,
        )
        fig = render_figure_to_tensor(fig)

        if is_main_process():
            logger = get_tensorboard_logger(runner)
            logger.add_image(f"{self.prefix}/epoch", fig, global_step=runner.global_epoch)


class F1ScoreCallback(Callback):
    """
    Compute F1 metric score

    """

    def __init__(
        self,
        num_classes: int,
        outputs_to_labels: Callable[[Tensor], Tensor] = argmax_over_dim_1,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f1",
        average="macro",
        ignore_index: Optional[int] = None,
        zero_division="warn",
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.num_classes = num_classes
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.outputs_to_labels = outputs_to_labels
        self.average = average
        self.confusion_matrix = None
        self.zero_division = zero_division

    def on_loader_start(self, state):
        self.confusion_matrix = np.zeros((self.num_classes, 2, 2), dtype=np.long)

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        pred_labels = self.outputs_to_labels(runner.output[self.output_key])
        true_labels = runner.input[self.input_key].type_as(pred_labels)

        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        if self.ignore_index is not None:
            mask = true_labels != self.ignore_index
            pred_labels = torch.masked_select(pred_labels, mask)
            true_labels = torch.masked_select(true_labels, mask)

        if len(true_labels):
            true_labels = to_numpy(true_labels)
            pred_labels = to_numpy(pred_labels)
            batch_cm = multilabel_confusion_matrix(
                y_true=true_labels, y_pred=pred_labels, labels=np.arange(self.num_classes, dtype=int)
            )
            self.confusion_matrix = self.confusion_matrix + batch_cm

    def on_loader_end(self, runner: IRunner):
        MCM = np.sum(all_gather(self.confusion_matrix), axis=0)
        metric = self._f1_from_confusion_matrix(MCM, average=self.average, zero_division=self.zero_division)
        runner.loader_metrics[self.prefix] = metric

    def _f1_from_confusion_matrix(
        self, MCM, average, beta=1, warn_for=("precision", "recall", "f-score"), zero_division="warn"
    ):
        """
        Code borrowed from sklear.metrics

        Args:
            MCM:
            average:
            beta:
            warn_for:
            zero_division:

        Returns:

        """
        tp_sum = MCM[:, 1, 1]
        pred_sum = tp_sum + MCM[:, 0, 1]
        true_sum = tp_sum + MCM[:, 1, 0]

        if average == "micro":
            tp_sum = np.array([tp_sum.sum()])
            pred_sum = np.array([pred_sum.sum()])
            true_sum = np.array([true_sum.sum()])

        # Finally, we have all our sufficient statistics. Divide! #
        beta2 = beta ** 2

        # Divide, and on zero-division, set scores and/or warn according to
        # zero_division:
        from sklearn.metrics._classification import _prf_divide, _warn_prf

        precision = _prf_divide(tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division)
        recall = _prf_divide(tp_sum, true_sum, "recall", "true", average, warn_for, zero_division)

        # warn for f-score only if zero_division is warn, it is in warn_for
        # and BOTH prec and rec are ill-defined
        if zero_division == "warn" and ("f-score",) == warn_for:
            if (pred_sum[true_sum == 0] == 0).any():
                _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

        # if tp == 0 F will be 1 only if all predictions are zero, all labels are
        # zero, and zero_division=1. In all other case, 0
        if np.isposinf(beta):
            f_score = recall
        else:
            denom = beta2 * precision + recall

            denom[denom == 0.0] = 1  # avoid division by 0
            f_score = (1 + beta2) * precision * recall / denom

        # Average the results
        if average == "weighted":
            weights = true_sum
            if weights.sum() == 0:
                zero_division_value = 0.0 if zero_division in ["warn", 0] else 1.0
                # precision is zero_division if there are no positive predictions
                # recall is zero_division if there are no positive labels
                # fscore is zero_division if all labels AND predictions are
                # negative
                return (
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0,
                    None,
                )
        else:
            weights = None

        if average is not None:
            assert average != "binary" or len(precision) == 1
            f_score = np.average(f_score, weights=weights)

        return f_score


def binary_dice_iou_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
    ignore_index=None,
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
    :param ignore_index:
    :return: Float scalar
    """
    assert mode in {"dice", "iou"}

    # Make binary predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    if ignore_index is not None:
        mask = (y_true != ignore_index).to(y_true.dtype)
        y_true = y_true * mask
        y_pred = y_pred * mask

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality - intersection + eps)

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
    ignore_index=None,
):
    ious = []
    num_classes = y_pred.size(0)
    y_pred = y_pred.argmax(dim=0)

    if classes_of_interest is None:
        classes_of_interest = range(num_classes)

    for class_index in classes_of_interest:
        y_pred_i = (y_pred == class_index).float()
        y_true_i = (y_true == class_index).float()
        if ignore_index is not None:
            not_ignore_mask = (y_true != ignore_index).float()
            y_pred_i *= not_ignore_mask
            y_true_i *= not_ignore_mask

        iou = binary_dice_iou_score(
            y_pred=y_pred_i,
            y_true=y_true_i,
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
    ignore_index=None,
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
            ignore_index=ignore_index,
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
        ignore_index=None,
    ):
        """
        :param mode: One of: 'binary', 'multiclass', 'multilabel'.
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        if mode not in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}:
            raise ValueError("Mode must be one of BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE")

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
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                mode=metric,
                ignore_index=ignore_index,
            )

        if self.mode == MULTICLASS_MODE:
            self.score_fn = partial(
                multiclass_dice_iou_score,
                mode=metric,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
                ignore_index=ignore_index,
            )

        if self.mode == MULTILABEL_MODE:
            self.score_fn = partial(
                multilabel_dice_iou_score,
                mode=metric,
                threshold=0.0,
                nan_score_on_empty=nan_score_on_empty,
                classes_of_interest=self.classes_of_interest,
                ignore_index=ignore_index,
            )

    def on_loader_start(self, state):
        self.scores = []

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        outputs = runner.output[self.output_key].detach()
        targets = runner.input[self.input_key].detach()

        batch_size = targets.size(0)
        score_per_image = []
        for image_index in range(batch_size):
            score_per_class = self.score_fn(y_pred=outputs[image_index], y_true=targets[image_index])
            score_per_image.append(score_per_class)

        mean_score = np.nanmean(score_per_image)
        runner.batch_metrics[self.prefix] = float(mean_score)
        self.scores.extend(score_per_image)

    def on_loader_end(self, runner: IRunner):
        scores = np.concatenate(all_gather(np.array(self.scores)))
        mean_per_class = np.nanmean(scores, axis=1)  # Average across classes
        mean_score = np.nanmean(mean_per_class, axis=0)  # Average across images

        runner.loader_metrics[self.prefix] = float(mean_score)

        # Log additional IoU scores per class
        if self.mode in {MULTICLASS_MODE, MULTILABEL_MODE}:
            num_classes = scores.shape[1]
            class_names = self.class_names
            if class_names is None:
                class_names = [f"class_{i}" for i in range(num_classes)]

            scores_per_class = np.nanmean(scores, axis=0)
            for class_name, score_per_class in zip(class_names, scores_per_class):
                runner.loader_metrics[self.prefix + "_" + class_name] = float(score_per_class)


class OutputDistributionCallback(Callback):
    """
    Plot histogram of predictions for each class. This callback supports binary & multi-classs predictions
    """

    def __init__(
        self, input_key: str, output_key: str, output_activation: Callable, num_classes: int, prefix="distribution"
    ):
        """

        Args:
            input_key:
            output_key:
            output_activation: A function that should convert logits to class labels
            For binary predictions this could be `lambda x: int(x > 0.5)` or `lambda x: torch.argmax(x, dim=1)`
            for multi-class predictions.
            num_classes: Number of classes. Must be 2 for binary.
            prefix:
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.true_labels = []
        self.pred_labels = []
        self.num_classes = num_classes
        self.output_activation = output_activation

    def on_loader_start(self, state: IRunner):
        self.true_labels = []
        self.pred_labels = []

    @torch.no_grad()
    def on_batch_end(self, state: IRunner):
        output = state.output[self.output_key].detach()
        self.true_labels.extend(to_numpy(state.input[self.input_key]).flatten())
        self.pred_labels.extend(to_numpy(self.output_activation(output)).flatten())

    def on_loader_end(self, state: IRunner):
        true_labels = np.concatenate(all_gather(np.array(self.true_labels)))
        pred_probas = np.concatenate(all_gather(np.array(self.pred_labels)))

        if is_main_process():
            logger = get_tensorboard_logger(state)

            for class_label in range(self.num_classes):
                logger.add_histogram(
                    f"{self.prefix}/{class_label}", pred_probas[true_labels == class_label], state.epoch
                )


class AccuracyCallback(Callback):
    """
    Accuracy metric callback.
    DDP mode supported
    """

    def __init__(
        self,
        outputs_to_labels: Callable[[Tensor], Tensor] = argmax_over_dim_1,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
            num_classes: number of classes to calculate ``topk_args``
                if ``accuracy_args`` is None
        """

        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.outputs_to_labels = outputs_to_labels
        self.correct = 0
        self.totals = 0

    def on_loader_start(self, state):
        self.correct = 0
        self.totals = 0

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        pred_labels = self.outputs_to_labels(runner.output[self.output_key])
        true_labels = runner.input[self.input_key].type_as(pred_labels)

        true_labels = true_labels.view(-1)
        pred_labels = pred_labels.view(-1)

        if self.ignore_index is not None:
            mask = true_labels != self.ignore_index
            pred_labels = torch.masked_select(pred_labels, mask)
            true_labels = torch.masked_select(true_labels, mask)

        batch_correct = int((pred_labels == true_labels).sum())
        batch_totals = len(true_labels)

        if len(true_labels):
            self.correct += batch_correct
            self.totals += batch_totals

        batch_accuracy = float(batch_correct) / float(batch_totals)
        runner.batch_metrics[self.prefix] = batch_accuracy

    def on_loader_end(self, runner: IRunner):
        correct = np.sum(all_gather(self.correct))
        total = np.sum(all_gather(self.totals))
        accuracy = float(correct) / float(total)
        runner.loader_metrics[self.prefix] = accuracy


class MultilabelAccuracyCallback(Callback):
    """
    Accuracy score metric for multi-label case (aka Exact Match Ratio, Subset accuracy).
    """

    def __init__(
        self,
        outputs_to_labels: Callable[[Tensor], Tensor] = argmax_over_dim_1,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        ignore_index: Optional[int] = None,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
            num_classes: number of classes to calculate ``topk_args``
                if ``accuracy_args`` is None
        """

        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.outputs_to_labels = outputs_to_labels
        self.correct = 0
        self.totals = 0

    def on_loader_start(self, state):
        self.correct = 0
        self.totals = 0

    @torch.no_grad()
    def on_batch_end(self, runner: IRunner):
        pred_labels = self.outputs_to_labels(runner.output[self.output_key])
        true_labels = runner.input[self.input_key].type_as(pred_labels)

        correct_preds = pred_labels == true_labels

        if self.ignore_index is not None:
            mask = true_labels == self.ignore_index
            correct_preds = correct_preds | mask

        batch_correct = correct_preds.all(dim=1, keepdim=False)
        if len(batch_correct.size()) > 1:
            batch_correct = batch_correct.view((batch_correct.size(0), -1)).mean(dim=1)

        batch_totals = int(batch_correct.numel())
        batch_correct = float(batch_correct.float().sum())

        if len(true_labels):
            self.correct += batch_correct
            self.totals += batch_totals

        batch_accuracy = float(batch_correct) / float(batch_totals)
        runner.batch_metrics[self.prefix] = batch_accuracy

    def on_loader_end(self, runner: IRunner):
        correct = np.sum(all_gather(self.correct))
        total = np.sum(all_gather(self.totals))
        accuracy = float(correct) / float(total)
        runner.loader_metrics[self.prefix] = accuracy
