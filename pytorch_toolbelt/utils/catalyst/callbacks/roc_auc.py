from typing import Callable, Optional

import numpy as np
import torch
from catalyst.core import Callback, CallbackOrder
from sklearn.metrics import roc_auc_score
from torch import Tensor

__all__ = ["RocAucMetricCallback"]

from pytorch_toolbelt.utils import to_numpy
from pytorch_toolbelt.utils.distributed import all_gather


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

    def on_loader_start(self, state):
        self.y_trues = []
        self.y_preds = []

    @torch.no_grad()
    def on_batch_end(self, runner):
        pred_probas = self.outputs_to_probas(runner.output[self.output_key])
        true_labels = runner.input[self.input_key]

        self.y_trues.extend(to_numpy(true_labels))
        self.y_preds.extend(to_numpy(pred_probas))

    def on_loader_end(self, runner):
        y_trues = np.concatenate(all_gather(self.y_trues))
        y_preds = np.concatenate(all_gather(self.y_preds))
        score = roc_auc_score(y_true=y_trues, y_score=y_preds, average=self.average)
        runner.loader_metrics[self.prefix] = float(score)
