import torch
import numpy as np
from catalyst.dl.callbacks import Callback, RunnerState, TensorboardLogger, MultiMetricCallback, MetricCallback
from sklearn.metrics import f1_score, confusion_matrix

from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from pytorch_toolbelt.utils.visualization import render_figure_to_tensor, plot_confusion_matrix
from tensorboardX import SummaryWriter


def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
    for logger in state.loggers:
        if isinstance(logger, TensorboardLogger):
            return logger.loggers[state.loader_name]
    raise RuntimeError(f"Cannot find Tensorboard logger for loader {state.loader_name}")


class ShowPolarBatchesCallback(Callback):
    def __init__(self, visualize_batch, metric: str = "loss", minimize: bool = True, min_delta: float = 1e-6):
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

    def _log_image(self, loggers, mode: str, image, name, step: int, suffix=""):
        for logger in loggers:
            if isinstance(logger, TensorboardLogger):
                logger.loggers[mode].add_image(f"{name}{suffix}", tensor_from_rgb_image(image), step)

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
            for i, image in enumerate(best_samples):
                logger.add_image(f"Best Batch/{i}/epoch", tensor_from_rgb_image(image), state.step)

        if self.worst_score is not None:
            worst_samples = self.visualize_batch(self.worst_input, self.worst_output)
            for i, image in enumerate(worst_samples):
                logger.add_image(f"Worst Batch/{i}/epoch", tensor_from_rgb_image(image), state.step)


class EpochJaccardMetric(Callback):
    """
    Jaccard metric callback which is computed across whole epoch, not per-batch.
    """

    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "jaccard"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.intersection = 0
        self.union = 0

    def on_loader_start(self, state):
        self.intersection = 0
        self.union = 0

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key].detach()
        targets = state.input[self.input_key].detach()

        # Binarize outputs as we don't want to compute soft-jaccard
        outputs = (outputs > 0).float()

        intersection = float(torch.sum(targets * outputs))
        union = float(torch.sum(targets) + torch.sum(outputs))
        self.intersection += intersection
        self.union += union

    def on_loader_end(self, state):
        metric_name = self.prefix
        eps = 1e-7
        metric = self.intersection / (self.union - self.intersection + eps)
        state.metrics.epoch_values[state.loader_name][metric_name] = metric

        logger = _get_tensorboard_logger(state)
        logger.add_scalar(f"{self.prefix}/epoch", metric, global_step=state.epoch)


def pixel_accuracy(outputs, targets):
    """Compute the pixel accuracy
    """
    outputs = (outputs.detach() > 0).float()

    correct = float(torch.sum(outputs == targets.detach()))
    total = targets.numel()
    return correct / total


class PixelAccuracyMetric(MetricCallback):
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


class EpochMacroF1Metric(Callback):
    """Macro F1 epoch-wise metric callback.
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

        logger = _get_tensorboard_logger(state)
        logger.add_scalar(f"{self.prefix}/epoch", metric, global_step=state.epoch)


class ConfusionMatrixCallback(Callback):
    """
    Macro F1 epoch-wise metric callback.
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
        self.metric_fn = lambda outputs, targets: f1_score(targets, outputs, average='macro')
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
        targets = self.targets
        outputs = self.outputs

        if self.class_names is None:
            class_names = [str(i) for i in range(targets.shape[1])]
        else:
            class_names = self.class_names

        num_classes = len(class_names)
        cm = confusion_matrix(outputs, targets, labels=range(num_classes))

        fig = plot_confusion_matrix(cm, class_names=class_names, normalize=True, noshow=True)
        fig = render_figure_to_tensor(fig)

        logger = _get_tensorboard_logger(state)
        logger.add_image(f'{self.prefix}/epoch', fig, global_step=state.step)


class MacroF1Callback(Callback):
    """Macro F1 epoch-wise metric callback.
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

        logger = _get_tensorboard_logger(state)
        logger.add_scalar(f'{self.prefix}/epoch', metric, global_step=state.step)
