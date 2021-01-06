from typing import List

import numpy as np
import torch
from catalyst.dl import IRunner, CriterionCallback, Callback, CallbackOrder

__all__ = ["MixupInputCallback", "MixupCriterionCallback"]


class MixupInputCallback(Callback):
    """
    Callback to do mixup augmentation.

    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and does its work.
        You may not use them together.
    """

    def __init__(self, fields: List[str] = ("features",), alpha=0.5, on_train_only=True, p=0.5, **kwargs):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(CallbackOrder.Internal)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.is_needed = True
        self.is_batch_needed = True
        self.p = p

    def on_loader_start(self, state: IRunner):
        self.is_needed = not self.on_train_only or state.loader_name.startswith("train")

    def on_batch_start(self, state: IRunner):
        if not self.is_needed:
            return

        is_batch_needed = np.random.random() < self.p

        if is_batch_needed:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        index = torch.randperm(state.input[self.fields[0]].shape[0]).to(state.device)
        state.input["mixup_index"] = index
        state.input["mixup_lambda"] = lam

        for f in self.fields:
            a = lam * state.input[f]
            b = (1 - lam) * state.input[f][index]
            state.input[f] = a + b


class MixupCriterionCallback(CriterionCallback):
    """
    Callback to do mixup augmentation.

    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """

    def __init__(self, on_train_only=True, **kwargs):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.is_needed = True

    def on_loader_start(self, state: IRunner):
        self.is_needed = not self.on_train_only or state.loader_name.startswith("train")

    def _compute_loss_value(self, state: IRunner, criterion):
        if not self.is_needed:
            return super()._compute_loss_value(state, criterion)

        lam = state.input["mixup_lambda"]
        index = state.input["mixup_index"]

        pred = self._get_output(state.output, self.output_key)
        y_a = self._get_input(state.input, self.input_key)
        y_b = y_a[index]

        loss = lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
        return loss

    def _compute_loss_key_value(self, state: IRunner, criterion):
        if not self.is_needed:
            return super()._compute_loss_key_value(state, criterion)

        lam = state.input["mixup_lambda"]
        index = state.input["mixup_index"]

        pred = self._get_output(state.output, self.output_key)
        y_a = self._get_input(state.input, self.input_key)
        y_b = y_a[index]

        loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        return loss
