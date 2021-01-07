import torch
from catalyst.callbacks import CriterionCallback
from catalyst.core import IRunner
from torch import nn, Tensor
from torch.nn.functional import mse_loss, smooth_l1_loss

__all__ = ["TrainOnlyCriterionCallback"]


class TrainOnlyCriterionCallback(CriterionCallback):
    def on_batch_end(self, runner: IRunner) -> None:
        if runner.is_train_loader:
            return super(TrainOnlyCriterionCallback, self).on_batch_end(runner)
        else:
            runner.batch_metrics[self.prefix] = torch.tensor(0, device="cuda")
