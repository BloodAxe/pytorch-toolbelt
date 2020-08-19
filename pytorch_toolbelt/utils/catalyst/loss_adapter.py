from typing import Dict

import torch
from catalyst.dl import (
    CriterionCallback,
    IRunner,
)
from torch import nn, Tensor

__all__ = [
    "TrainOnlyCriterionCallback",
    "PassthroughCriterionCallback",
    "LossModule",
    "LossWrapper",
]


class TrainOnlyCriterionCallback(CriterionCallback):
    """
    Computes loss only on training stage
    """

    def _compute_loss_value(self, state: IRunner, criterion):
        predictions = self._get_output(state.output, self.output_key)
        targets = self._get_input(state.input, self.input_key)

        if state.loader_name != "train":
            return torch.tensor(0, device=predictions.device, dtype=predictions.dtype)

        loss = criterion(predictions, targets)
        return loss

    def _compute_loss_key_value(self, state: IRunner, criterion):
        output = self._get_output(state.output, self.output_key)
        input = self._get_input(state.input, self.input_key)

        if state.loader_name != "train":
            return torch.tensor(0, device=output.device, dtype=output.dtype)

        loss = criterion(**output, **input)
        return loss


class PassthroughCriterionCallback(CriterionCallback):
    """
    Returns one of model's outputs as loss values
    """

    def __init__(self, output_key, multiplier=1.0):
        super().__init__(output_key=output_key, prefix=output_key, multiplier=multiplier)

    def _compute_loss_value(self, state: IRunner, criterion):
        loss = self._get_output(state.output, self.output_key)
        return loss.mean()

    def _compute_loss_key_value(self, state: IRunner, criterion):
        loss = self._get_output(state.output, self.output_key)
        return loss.mean()


class LossModule(nn.Module):
    def __init__(self, output_key: str, target_key: str, loss_fn):
        super().__init__()
        self.output_key = output_key
        self.target_ley = target_key
        self.loss_fn = loss_fn

    def forward(self, outputs, targets):  # skipcq: PYL-W0221
        return self.loss_fn(outputs[self.output_key], targets[self.target_ley])


class LossWrapper(nn.Module):
    """
    A wrapper module around model that computes one or many loss functions and extends output dictionary with
    their values. The point of this wrapper is that loss computed on each GPU node in parallel.

    Usage:
    >>> from catalyst.dl import SupervisedRunner
    >>> runner = SupervisedRunner(input_key=None, output_key=None, device="cuda")
    >>> runner._default_experiment = ParallelLossSupervisedExperiment
    >>> loss_modules = {
    >>>     "my_loss": LossModule(
    >>>         output_key="logits",
    >>>         target_key="targets",
    >>>         loss_fn=nn.BCEWithLogitsLoss(),
    >>>     )}
    >>> loss_callback = PassthroughCriterionCallback("my_loss")
    >>> runner.train(
    >>>     callbacks=[loss_callback, ...]
    >>>     model=LossWrapper(model, "image", loss_modules),
    >>>     ...)

    Note, that SupervisedRunner adds default CriterionCallback
    """

    def __init__(self, model: nn.Module, input_key: str, losses: Dict[str, LossModule]):
        super().__init__()
        self.model = model
        self.input_key = input_key
        self.loss_names = list(losses.keys())
        self.losses = nn.ModuleList([losses[key] for key in self.loss_names])

    def forward(self, **input: Dict[str, Tensor]) -> Dict[str, Tensor]:  # skipcq: PYL-W0221
        output: Dict[str, Tensor] = self.model(input[self.input_key])

        for output_loss_key, loss in zip(self.loss_names, self.losses):
            output[output_loss_key] = loss(output, input)

        return output

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.model.state_dict()
