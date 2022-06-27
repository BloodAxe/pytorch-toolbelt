import math
import warnings
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    LambdaLR,
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

__all__ = [
    "GradualWarmupScheduler",
    "FlatCosineAnnealingLR",
    "OnceCycleLR",
    "CosineAnnealingLRWithDecay",
    "CosineAnnealingWarmRestartsWithDecay",
    "PolyLR",
]


def set_learning_rate(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


class OnceCycleLR(_LRScheduler):
    def __init__(self, optimizer, epochs, min_lr_factor=0.05, max_lr=1.0):
        half_epochs = epochs // 2
        decay_epochs = int(epochs * 0.05)

        lr_grow = np.linspace(min_lr_factor, max_lr, num=half_epochs)
        lr_down = np.linspace(max_lr, min_lr_factor, num=int(epochs - half_epochs - decay_epochs))
        lr_decay = np.linspace(min_lr_factor, min_lr_factor * 0.01, int(decay_epochs))
        self.learning_rates = np.concatenate((lr_grow, lr_down, lr_decay)) / max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.learning_rates[self.last_epoch] for base_lr in self.base_lrs]


class CosineAnnealingLRWithDecay(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

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

    def __init__(self, optimizer, T_max, gamma, eta_min=0, last_epoch=-1):
        self.gamma = gamma
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLRWithDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        def compute_lr(base_lr):
            return (
                self.eta_min
                + (base_lr * self.gamma**self.last_epoch - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / 2
            )

        return [compute_lr(base_lr) for base_lr in self.base_lrs]


class PolyLR(LambdaLR):
    def __init__(self, optimizer: Optimizer, max_epoch, gamma=0.9):
        def poly_lr(epoch):
            return (1.0 - float(epoch) / max_epoch) ** gamma

        super().__init__(optimizer, poly_lr)


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
            + (base_lr * self.gamma**self.last_epoch - self.eta_min)
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
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
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


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.use("module://backend_interagg")
    import matplotlib.pyplot as plt

    from torch.optim import SGD, Optimizer

    net = nn.Conv2d(1, 1, 1)
    opt = SGD(net.parameters(), lr=1e-3)

    epochs = 100

    plt.figure()

    scheduler = OnceCycleLR(opt, epochs + 1, min_lr_factor=0.01)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="1cycle")

    scheduler = CosineAnnealingLRWithDecay(opt, epochs / 5, gamma=0.99)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="cosine")

    scheduler = PolyLR(opt, epochs, gamma=0.9)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="poly")

    scheduler = FlatCosineAnnealingLR(opt, epochs, T_flat=epochs // 2, eta_min=1e-6)
    lrs = []
    for epoch in range(epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    plt.plot(range(epochs), lrs, label="flat_cos")

    plt.legend()
    plt.show()
