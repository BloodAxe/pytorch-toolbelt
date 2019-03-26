import math
import numpy as np
from torch import nn

from torch.optim.lr_scheduler import _LRScheduler


def set_learning_rate(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


class OnceCycleLR(_LRScheduler):
    def __init__(self, optimizer, epochs, min_lr_factor=0.05, max_lr=1.):
        half_epochs = epochs // 2
        decay_epochs = (epochs * 0.05)

        lr_grow = np.linspace(min_lr_factor, max_lr, half_epochs)
        lr_down = np.linspace(max_lr, min_lr_factor, half_epochs - decay_epochs)
        lr_decay = np.linspace(min_lr_factor, min_lr_factor * 0.01, decay_epochs)
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
            return self.eta_min + (base_lr * self.gamma ** self.last_epoch - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2

        return [compute_lr(base_lr) for base_lr in self.base_lrs]


if __name__ == '__main__':
    import matplotlib as mpl

    mpl.use('module://backend_interagg')
    import matplotlib.pyplot as plt

    from torch.optim import SGD

    net = nn.Conv2d(1, 1, 1)
    opt = SGD(net.parameters(), lr=1e-2)

    scheduler = OnceCycleLR(opt, 800, min_lr_factor=0.01)
    # scheduler = CosineAnnealingLRWithDecay(opt, 80, gamma=0.999)

    lrs = []
    for epoch in range(800):
        scheduler.step(epoch)
        lrs.append(scheduler.get_lr()[0])

    plt.figure()
    plt.plot(range(800), lrs)
    plt.show()
