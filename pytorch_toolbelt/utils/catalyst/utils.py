import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import safitty
import torch
from catalyst import utils
from catalyst.dl import RunnerState, Callback, CallbackOrder
from catalyst.dl.callbacks.checkpoint import BaseCheckpointCallback

__all__ = ["clean_checkpoint", "report_checkpoint", "BestMetricCheckpointCallback", "HyperParametersCallback"]

from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger


def clean_checkpoint(src_fname, dst_fname):
    """
    Remove optimizer, scheduler and criterion states from checkpoint
    :param src_fname: Source checkpoint filename
    :param dst_fname: Target checkpoint filename (can be same)
    """
    checkpoint = torch.load(src_fname, map_location="cpu")

    keys = ["criterion_state_dict", "optimizer_state_dict", "scheduler_state_dict"]

    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, dst_fname)


def report_checkpoint(checkpoint: Dict):
    """
    Print checkpoint metrics and epoch number
    :param checkpoint:
    """
    print("Epoch          :", checkpoint["epoch"])

    skip_fields = [
        "_base/lr",
        "_base/momentum",
        "_timers/data_time",
        "_timers/model_time",
        "_timers/batch_time",
        "_timers/_fps",
    ]
    print(
        "Metrics (Train):", [(k, v) for k, v, in checkpoint["epoch_metrics"]["train"].items() if k not in skip_fields]
    )
    print(
        "Metrics (Valid):", [(k, v) for k, v, in checkpoint["epoch_metrics"]["valid"].items() if k not in skip_fields]
    )


class BestMetricCheckpointCallback(BaseCheckpointCallback):
    """
    Checkpoint callback to save model weights based on user-defined metric value.
    """

    def __init__(
        self,
        target_metric: str,
        target_metric_minimize=False,
        save_n_best: int = 3,
        checkpoints_dir=None,
        metric_filename: str = "_metrics.json",
    ):
        """
        Args:
            target_metric (str): name of the target metric to monitor.
            target_metric_minimize (bool): define whether metric is minimized.
            save_n_best (int): number of best checkpoint to keep
            checkpoints_dir (str): path to directory where checkpoints will be saved
            metric_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        if checkpoints_dir is None:
            checkpoints_dir = "checkpoints_" + target_metric

        super().__init__(metric_filename)
        self.main_metric = target_metric
        self.minimize_metric = target_metric_minimize
        self.save_n_best = save_n_best
        self.top_best_metrics = []
        self.epochs_metrics = []
        self.checkpoints_dir = checkpoints_dir
        self.best_main_metric_value = None

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        result = f"{checkpoint['stage']}.{checkpoint['epoch']}"
        return result

    def get_metric(self, last_valid_metrics) -> Dict:
        top_best_checkpoints = [
            (Path(filepath).stem, valid_metric) for (filepath, _, valid_metric) in self.top_best_metrics
        ]
        all_epochs_metrics = [
            (f"epoch_{order_index}", valid_metric) for (order_index, valid_metric) in enumerate(self.epochs_metrics)
        ]
        best_valid_metrics = top_best_checkpoints[0][1]
        metrics = OrderedDict(
            [("best", best_valid_metrics)] + [("last", last_valid_metrics)] + top_best_checkpoints + all_epochs_metrics
        )

        self.metrics = metrics
        return self.metrics

    def truncate_checkpoints(self, minimize_metric: bool) -> None:
        self.top_best_metrics = sorted(self.top_best_metrics, key=lambda x: x[1], reverse=not minimize_metric)
        if len(self.top_best_metrics) > self.save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = Path(last_item[0])
            last_filepaths = last_filepath.parent.glob(last_filepath.name.replace(".pth", "*"))
            for filepath in last_filepaths:
                os.remove(filepath)

    def process_checkpoint(
        self, logdir: str, checkpoint: Dict, is_best: bool, main_metric: str = "loss", minimize_metric: bool = True
    ):
        suffix = self.get_checkpoint_suffix(checkpoint)
        utils.save_checkpoint(
            logdir=Path(logdir) / Path(self.checkpoints_dir),
            checkpoint=checkpoint,
            suffix=f"{suffix}_full",
            is_best=is_best,
            is_last=True,
            special_suffix="_full",
        )

        exclude = ["criterion", "optimizer", "scheduler"]
        checkpoint = {key: value for key, value in checkpoint.items() if all(z not in key for z in exclude)}
        filepath = utils.save_checkpoint(
            checkpoint=checkpoint,
            logdir=Path(logdir) / Path(self.checkpoints_dir),
            suffix=suffix,
            is_best=is_best,
            is_last=True,
        )

        valid_metrics = checkpoint["valid_metrics"]
        checkpoint_metric = valid_metrics[main_metric]
        metrics_record = (filepath, checkpoint_metric, valid_metrics)
        self.top_best_metrics.append(metrics_record)
        self.epochs_metrics.append(metrics_record)
        self.truncate_checkpoints(minimize_metric=minimize_metric)
        metrics = self.get_metric(valid_metrics)
        self.save_metric(logdir, metrics)

    def on_stage_start(self, state: RunnerState):
        self.best_main_metric_value: float = float("+inf") if self.minimize_metric else float("-inf")

    def on_epoch_end(self, state: RunnerState):
        if state.stage.startswith("infer"):
            return

        valid_metrics = dict(state.metrics.valid_values)
        epoch_metrics = dict(state.metrics.epoch_values)

        checkpoint = utils.pack_checkpoint(
            model=state.model,
            criterion=state.criterion,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            epoch_metrics=epoch_metrics,
            valid_metrics=valid_metrics,
            stage=state.stage,
            epoch=state.epoch_log,
            checkpoint_data=state.checkpoint_data,
        )

        main_metric_value = valid_metrics[self.main_metric]
        if self.minimize_metric:
            is_best = main_metric_value < self.best_main_metric_value
        else:
            is_best = main_metric_value > self.best_main_metric_value

        if is_best:
            self.best_main_metric_value = main_metric_value

        self.process_checkpoint(
            logdir=state.logdir,
            checkpoint=checkpoint,
            is_best=is_best,
            main_metric=self.main_metric,
            minimize_metric=self.minimize_metric,
        )

    def on_stage_end(self, state: RunnerState):
        print("Top best models:")
        top_best_metrics_str = "\n".join(
            [
                "{filepath}\t{metric:3.4f}".format(filepath=filepath, metric=checkpoint_metric)
                for filepath, checkpoint_metric, _ in self.top_best_metrics
            ]
        )
        print(top_best_metrics_str)

    def save_metric(self, logdir: str, metrics: Dict) -> None:
        safitty.save(metrics, f"{logdir}/{self.checkpoints_dir}/{self.metric_filename}")

    def on_exception(self, state: RunnerState):
        exception = state.exception
        if not utils.is_exception(exception):
            return

        try:
            valid_metrics = state.metrics.valid_values
            epoch_metrics = state.metrics.epoch_values
            checkpoint = utils.pack_checkpoint(
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                epoch_metrics=epoch_metrics,
                valid_metrics=valid_metrics,
                stage=state.stage,
                epoch=state.epoch_log,
                checkpoint_data=state.checkpoint_data,
            )
            suffix = self.get_checkpoint_suffix(checkpoint)
            suffix = f"{suffix}.exception_{exception.__class__.__name__}"
            utils.save_checkpoint(
                logdir=Path(f"{state.logdir}/{self.checkpoints_dir}/"),
                checkpoint=checkpoint,
                suffix=suffix,
                is_best=False,
                is_last=False,
            )
            metrics = self.metrics
            metrics[suffix] = valid_metrics
            self.save_metric(state.logdir, metrics)
        except Exception:
            pass


class HyperParametersCallback(Callback):
    """
    Callback that logs hyper-parameters for training session and target metric value.
    Useful for evaluation of several runs in Tensorboard.
    """

    def __init__(self, hparam_dict: Dict):
        if "stage" in hparam_dict:
            raise KeyError("Key 'stage' is reserved")

        super().__init__(CallbackOrder.Metric)
        self.hparam_dict = hparam_dict

    def on_stage_end(self, state: RunnerState):
        logger = get_tensorboard_logger(state)

        hparam_dict = self.hparam_dict.copy()
        hparam_dict["stage"] = state.stage

        logger.add_hparams(
            hparam_dict=self.hparam_dict,
            metric_dict={"best_" + state.main_metric: state.metrics.best_main_metric_value},
        )
