import math

import torch
from catalyst.dl import IRunner, CriterionCallback
from torch import nn
from torch.nn import functional as F

__all__ = ["LPRegularizationCallback", "TSACriterionCallback", "get_multiplier"]


def get_multiplier(training_progress, schedule, start, end):
    if schedule is None or schedule == "none":
        threshold = 0
    elif schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = math.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - math.exp((-training_progress) * scale)
    else:
        raise KeyError(schedule)

    return threshold * (end - start) + start


class LPRegularizationCallback(CriterionCallback):
    """
    Generalized L1/L2 weight decay callback that may exponentially grow
    """

    def __init__(
        self,
        on_train_only=True,
        apply_to_bias=False,
        prefix: str = None,
        p=1,
        start_wd=0,
        end_wd=1e-4,
        schedule="exp_schedule",
    ):
        """

        :param on_train_only:
        :param apply_to_bias:
        :param prefix:
        :param p:
        :param start_wd:
        :param end_wd:
        :param schedule:
        """
        if prefix is None:
            prefix = f"l{self.p}_loss"

        super().__init__(prefix=prefix, multiplier=start_wd)

        self.on_train_only = on_train_only
        self.is_needed = True
        self.apply_to_bias = apply_to_bias
        self.schedule = schedule
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.p = p
        self.multiplier = None

    def on_loader_start(self, runner: IRunner):
        self.is_needed = not self.on_train_only or runner.loader_name.startswith("train")
        if self.is_needed:
            runner.metrics.epoch_values[runner.loader_name][f"l{self.p}_weight_decay"] = self.multiplier

    def on_epoch_start(self, runner: IRunner):
        training_progress = float(runner.epoch) / float(runner.num_epochs)
        self.multiplier = get_multiplier(training_progress, self.schedule, self.start_wd, self.end_wd)

    def on_batch_end(self, runner: IRunner):
        if not self.is_needed:
            return

        lp_reg = 0

        for module in runner.model.children():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm1d, nn.BatchNorm3d)):
                continue

            for param_name, param in module.named_parameters():
                if param_name.endswith("bias") and not self.apply_to_bias:
                    continue

                if param.requires_grad:
                    lp_reg = param.norm(self.p) * self.multiplier + lp_reg

        runner.batch_metrics.update(**{self.prefix: lp_reg.item()})


class TSACriterionCallback(CriterionCallback):
    """
    Criterion callback with training signal annealing support.

    This callback requires that criterion key returns loss per each element in batch

    Reference:
        Unsupervised Data Augmentation for Consistency Training
        https://arxiv.org/abs/1904.12848
    """

    def __init__(
        self,
        num_classes,
        num_epochs,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0,
        ignore_index=-100,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            multiplier=multiplier,
        )
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.tsa_threshold = None
        self.ignore_index = ignore_index

    def get_tsa_threshold(self, current_epoch, schedule, start, end) -> float:
        training_progress = float(current_epoch) / float(self.num_epochs)
        threshold = None
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
            # [exp(-5), exp(0)] = [1e-2, 1]
        elif schedule == "log_schedule":
            scale = 5
            # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
            threshold = 1 - math.exp((-training_progress) * scale)
        else:
            raise KeyError(f"Unsupported schedule name {schedule}")
        return threshold * (end - start) + start

    def on_epoch_start(self, runner: IRunner):
        if runner.loader_name == "train":
            self.tsa_threshold = self.get_tsa_threshold(runner.epoch, "exp_schedule", 1.0 / self.num_classes, 1.0)
            runner.epoch_metrics["train"]["tsa_threshold"] = self.tsa_threshold

    def _compute_loss(self, runner: IRunner, criterion):

        logits = runner.output[self.output_key]
        targets = runner.input[self.input_key]
        supervised_mask = targets != self.ignore_index  # Mask indicating labeled samples

        targets = targets[supervised_mask]
        logits = logits[supervised_mask]

        if not len(targets):
            return torch.tensor(0, dtype=logits.dtype, device=logits.device)

        with torch.no_grad():
            one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).to(logits.dtype)
            sup_probs = logits.detach().softmax(dim=1)
            correct_label_probs = torch.sum(one_hot_targets * sup_probs, dim=1)
            larger_than_threshold = correct_label_probs > self.tsa_threshold
            loss_mask = 1.0 - larger_than_threshold.to(logits.dtype)

        loss = criterion(logits, targets)
        loss = loss * loss_mask

        loss = loss.sum() / loss_mask.sum().clamp_min(1)
        return loss
