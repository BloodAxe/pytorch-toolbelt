import math
from typing import List, Optional, Union, Any

import torch
import numpy as np
from catalyst.dl import IRunner, Callback, CallbackOrder

__all__ = ["StopIfNanCallback"]

from torch import Tensor


def _any_is_nan(x: Union[Tensor, np.ndarray, float, List[Any], Any]) -> bool:
    if torch.is_tensor(x):
        return x.isnan().any()
    elif isinstance(x, np.ndarray):
        return np.isnan(x).any()
    elif isinstance(x, (list, tuple)):
        for e in x:
            if _any_is_nan(e):
                return True
    elif isinstance(x, float):
        return math.isnan(x)

    return False


class StopIfNanCallback(Callback):
    """
    Stop training process if NaN observed in batch_metrics
    """

    def __init__(self, metrics_to_monitor: Optional[List[str]] = None):
        super().__init__(CallbackOrder.Metric + 1)
        self.metrics_to_monitor = metrics_to_monitor

    def on_batch_end(self, runner: IRunner):
        if self.metrics_to_monitor is not None:
            keys = self.metrics_to_monitor
        else:
            keys = runner.batch_metrics.keys()

        for key in keys:
            if _any_is_nan(runner.batch_metrics[key]):
                print(
                    f"Stopping training due to NaN presence in {key} metric at epoch {runner.global_epoch}."
                    f"batch_metrics={{{runner.batch_metrics}}}"
                )
                runner.need_early_stop = True
