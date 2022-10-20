import torch
from catalyst.core import IRunner
from catalyst.callbacks import CriterionCallback

__all__ = ["TrainOnlyCriterionCallback"]


class TrainOnlyCriterionCallback(CriterionCallback):
    def on_batch_end(self, runner: IRunner) -> None:
        if runner.is_train_loader:
            return super(TrainOnlyCriterionCallback, self).on_batch_end(runner)
        else:
            runner.batch_metrics[self.prefix] = torch.tensor(0, device="cuda")
