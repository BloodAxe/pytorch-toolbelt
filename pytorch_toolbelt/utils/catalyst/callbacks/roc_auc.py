from typing import Callable, Optional

import numpy as np
import torch
from catalyst.core import Callback, CallbackOrder
from sklearn.metrics import roc_auc_score
from torch import Tensor

__all__ = ["RocAucMetricCallback"]

from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.distributed import all_gather, is_main_process
from pytorch_toolbelt.utils.catalyst.visualization import get_tensorboard_logger


class RocAucMetricCallback(Callback):
    """
    Roc Auc score metric
    """

    def __init__(
        self,
        outputs_to_probas: Callable[[Tensor], Tensor] = torch.sigmoid,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "roc_auc",
        average="macro",
        ignore_index: Optional[int] = None,
        log_pr_curve: bool = True,
        fix_nans=False,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.ignore_index = ignore_index
        self.outputs_to_probas = outputs_to_probas
        self.y_trues = []
        self.y_preds = []
        self.average = average
        self.log_pr_curve = log_pr_curve
        self.fix_nans = fix_nans

    def on_loader_start(self, state):
        self.y_trues = []
        self.y_preds = []

    @torch.no_grad()
    def on_batch_end(self, runner):
        pred_probas = self.outputs_to_probas(runner.output[self.output_key].float())
        true_labels = runner.input[self.input_key].float()

        # Aggregate flattened labels
        self.y_trues.extend(to_numpy(true_labels).reshape(-1))
        self.y_preds.extend(to_numpy(pred_probas).reshape(-1))

    def on_loader_end(self, runner):
        y_trues = np.concatenate(all_gather(self.y_trues))
        y_preds = np.concatenate(all_gather(self.y_preds))

        if self.fix_nans:
            y_preds[~np.isfinite(y_preds)] = 0.5

        score = roc_auc_score(y_true=y_trues, y_score=y_preds, average=self.average)
        runner.loader_metrics[self.prefix] = float(score)

        if self.log_pr_curve and is_main_process():
            logger = get_tensorboard_logger(runner)
            logger.add_pr_curve(
                self.prefix, predictions=y_preds, labels=y_trues, global_step=runner.global_epoch, num_thresholds=255
            )
