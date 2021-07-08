import math
import warnings

from catalyst.contrib.nn import OneCycleLRWithWarmup
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

__all__ = ["get_scheduler", "FlatCosineAnnealingLR", "CosineAnnealingWarmRestartsWithDecay"]


def get_scheduler(scheduler_name: str, optimizer, learning_rate: float, num_epochs: int, batches_in_epoch=None):
    if scheduler_name is None:
        name = ""
    else:
        name = scheduler_name.lower()

    need_warmup = "warmup_" in name
    name = name.replace("warmup_", "")

    scheduler = None

    if name == "cos":
        scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-6)
    elif name == "cos2":
        scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=float(learning_rate * 0.5))
    elif name == "cos10":
        scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=float(learning_rate * 0.1))
    elif name == "cosr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 4), eta_min=1e-6)
    elif name == "cosrd":
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer, T_0=max(2, num_epochs // 6), gamma=0.96, eta_min=1e-6
        )
    elif name in {"1cycle", "one_cycle"}:
        scheduler = OneCycleLRWithWarmup(
            optimizer,
            lr_range=(learning_rate, 1e-6),
            num_steps=batches_in_epoch * num_epochs,
            warmup_fraction=0.05,
            decay_fraction=0.1,
        )
    elif name == "exp":
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif name == "clr":
        scheduler = CyclicLR(
            optimizer,
            base_lr=1e-6,
            max_lr=learning_rate,
            step_size_up=batches_in_epoch // 4,
            # mode='exp_range',
            gamma=0.99,
        )
    elif name == "multistep":
        scheduler = MultiStepLR(
            optimizer, milestones=[int(num_epochs * 0.5), int(num_epochs * 0.7), int(num_epochs * 0.9)], gamma=0.3
        )
    elif name == "simple":
        scheduler = MultiStepLR(optimizer, milestones=[int(num_epochs * 0.4), int(num_epochs * 0.7)], gamma=0.1)
    else:
        raise KeyError(f"Unsupported scheduler name {name}")

    if need_warmup:
        scheduler = GradualWarmupScheduler(optimizer, 1.0, 5, after_scheduler=scheduler)
        print("Adding warmup")

    return scheduler
