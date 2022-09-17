from typing import Mapping, Any

from catalyst.contrib.nn import OneCycleLRWithWarmup
from pytorch_toolbelt.optimization.functional import get_lr_decay_parameters, get_optimizable_parameters
from pytorch_toolbelt.optimization.lr_schedules import (
    GradualWarmupScheduler,
    CosineAnnealingWarmRestartsWithDecay,
)
from pytorch_toolbelt.utils import master_print
from pytorch_toolbelt.utils.catalyst.pipeline import get_optimizer_cls
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CyclicLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
)

__all__ = ["get_scheduler", "get_optimizer"]


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    optimizer_params: Mapping[str, Any],
    apply_weight_decay_to_bias: bool = True,
    layerwise_params=None,
) -> Optimizer:
    """
    Construct an Optimizer for given model
    Args:
        model: Model to optimize. Only parameters that require_grad will be used
        optimizer_name: Name of the optimizer (case-insensitive). Supports native pytorch optimizers, apex and
            optimizers from pytorch-optimizers package.
        optimizer_params: Dict of optimizer params (lr, weight_decay, eps, etc)
        apply_weight_decay_to_bias: Whether to apply weight decay on bias parameters. Default is True
    Returns:
        Optimizer instance
    """

    # Optimizer parameter groups
    if layerwise_params is not None:
        if not apply_weight_decay_to_bias:
            raise ValueError("Layerwise params and no wd on bias are mutually exclusive")

        parameters = get_lr_decay_parameters(model, optimizer_params["lr"], layerwise_params)
    else:
        if apply_weight_decay_to_bias:
            parameters = get_optimizable_parameters(model)
        else:
            default_pg, biases_pg = [], []

            for k, v in model.named_parameters():
                if v.requires_grad:
                    if str.endswith(k, ".bias"):
                        biases_pg.append(v)  # biases
                    else:
                        default_pg.append(v)  # all else

            if apply_weight_decay_to_bias:
                parameters = default_pg + biases_pg
            else:
                parameters = default_pg

    optimizer_cls = get_optimizer_cls(optimizer_name)
    optimizer: Optimizer = optimizer_cls(
        parameters,
        **optimizer_params,
    )

    if not apply_weight_decay_to_bias:
        optimizer.add_param_group({"params": biases_pg, "weight_decay": 0.0})

    return optimizer


def get_scheduler(
    optimizer,
    name: str,
    learning_rate: float,
    num_epochs: int,
    batches_in_epoch=None,
    min_learning_rate: float = 1e-6,
    milestones=None,
    **kwargs,
):
    need_warmup = "warmup_" in name
    name = name.replace("warmup_", "")

    if name == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_learning_rate, **kwargs)
    elif name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, **kwargs)
    elif name == "cosr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 4), eta_min=min_learning_rate)
    elif name == "cosrd":
        scheduler = CosineAnnealingWarmRestartsWithDecay(
            optimizer, T_0=max(2, num_epochs // 6), eta_min=min_learning_rate
        )
    elif name in {"1cycle", "one_cycle"}:
        scheduler = OneCycleLRWithWarmup(
            optimizer,
            lr_range=(learning_rate, min_learning_rate),
            num_steps=batches_in_epoch * num_epochs,
            **kwargs,
        )
    elif name == "exp":
        scheduler = ExponentialLR(optimizer, **kwargs)
    elif name == "clr":
        scheduler = CyclicLR(
            optimizer,
            base_lr=min_learning_rate,
            max_lr=learning_rate,
            step_size_up=batches_in_epoch // 4,
            # mode='exp_range',
            gamma=0.99,
        )
    elif name == "multistep":
        milestones = [int(num_epochs * m) for m in milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, **kwargs)
    elif name == "simple":
        scheduler = MultiStepLR(optimizer, milestones=[int(num_epochs * 0.5), int(num_epochs * 0.8)], **kwargs)
    else:
        raise KeyError(f"Unsupported scheduler name {name}")

    if need_warmup:
        scheduler = GradualWarmupScheduler(optimizer, 1.0, 5, after_scheduler=scheduler)
        master_print("Adding warmup")

    return scheduler
