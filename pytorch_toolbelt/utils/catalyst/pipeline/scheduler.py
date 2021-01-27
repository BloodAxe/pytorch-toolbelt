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


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [max(1e-6, base_lr * (float(self.last_epoch) / self.total_epoch)) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class FlatCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        T_{cur} \neq (2k+1)T_{max};\\
        \eta_{t+1} = \eta_{t} + (\eta_{max} - \eta_{min})\frac{1 -
        \cos(\frac{1}{T_{max}}\pi)}{2},
        T_{cur} = (2k+1)T_{max}.\\
    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_flat, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_flat = T_flat
        self.eta_min = eta_min
        super(FlatCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                DeprecationWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs
        elif (max(0, self.last_epoch - self.T_flat) - 1 - max(0, self.T_max - self.T_flat)) % (
            2 * max(0, self.T_max - self.T_flat)
        ) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / max(0, self.T_max - self.T_flat))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * max(0, self.last_epoch - self.T_flat) / max(0, self.T_max - self.T_flat)))
            / (1 + math.cos(math.pi * (max(0, self.last_epoch - self.T_flat) - 1) / max(0, self.T_max - self.T_flat)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * max(0, self.last_epoch - self.T_flat) / max(0, self.T_max - self.T_flat)))
            / 2
            for base_lr in self.base_lrs
        ]


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
