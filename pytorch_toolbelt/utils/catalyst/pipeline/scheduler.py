import math
import warnings

from catalyst.contrib.nn import OneCycleLRWithWarmup
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

__all__ = ["get_scheduler", "CosineAnnealingWarmRestartsWithDecay"]


class CosineAnnealingWarmRestartsWithDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, gamma=0.9):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.gamma = gamma

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                DeprecationWarning,
            )

        return [
            self.eta_min
            + (base_lr * self.gamma ** self.last_epoch - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]


def get_scheduler(scheduler_name: str, optimizer, lr, num_epochs, batches_in_epoch=None):
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    if scheduler_name.lower() == "cos":
        return CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)

    if scheduler_name.lower() == "cos2":
        return CosineAnnealingLR(optimizer, num_epochs, eta_min=float(lr * 0.1))

    if scheduler_name.lower() == "cosr":
        return CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 4), eta_min=1e-6)

    if scheduler_name.lower() == "cosrd":
        return CosineAnnealingWarmRestartsWithDecay(optimizer, T_0=max(2, num_epochs // 6), gamma=0.96, eta_min=1e-6)

    if scheduler_name.lower() in {"1cycle", "one_cycle"}:
        return OneCycleLRWithWarmup(
            optimizer,
            lr_range=(lr, 1e-6),
            num_steps=batches_in_epoch * num_epochs,
            warmup_fraction=0.05,
            decay_fraction=0.1,
        )

    if scheduler_name.lower() == "exp":
        return ExponentialLR(optimizer, gamma=0.95)

    if scheduler_name.lower() == "clr":
        return CyclicLR(
            optimizer,
            base_lr=1e-6,
            max_lr=lr,
            step_size_up=batches_in_epoch // 4,
            # mode='exp_range',
            gamma=0.99,
        )

    if scheduler_name.lower() == "multistep":
        return MultiStepLR(
            optimizer, milestones=[int(num_epochs * 0.5), int(num_epochs * 0.7), int(num_epochs * 0.9)], gamma=0.3
        )

    if scheduler_name.lower() == "simple":
        return MultiStepLR(optimizer, milestones=[int(num_epochs * 0.4), int(num_epochs * 0.7)], gamma=0.1)

    raise KeyError(scheduler_name)
