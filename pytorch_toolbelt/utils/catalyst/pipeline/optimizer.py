from typing import Type, Dict, Any

from torch.optim.optimizer import Optimizer
from torch import nn
import torch.optim
import copy


__all__ = ["get_optimizer", "get_optimizer_cls", "scale_learning_rate_for_ddp"]


def get_optimizer_cls(optimizer_name) -> Type[Optimizer]:
    _OPTIMIZERS_REGISTRY = {
        "sgd": torch.optim.SGD,
        "asgd": torch.optim.ASGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "adamax": torch.optim.Adamax,
        "adagrad": torch.optim.Adagrad,
        "adadelta": torch.optim.Adadelta,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in _OPTIMIZERS_REGISTRY:
        return _OPTIMIZERS_REGISTRY[optimizer_name]

    _APEX_OPTIMIZERS = ["fused_adam", "fused_sgd", "fused_lamb"]
    if optimizer_name in _APEX_OPTIMIZERS:
        try:
            from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB

            _APEX_REGISTRY = {"fused_adam": FusedAdam, "fused_sgd": FusedSGD, "fused_lamb": FusedLAMB}
            if optimizer_name in _OPTIMIZERS_REGISTRY:
                return _APEX_REGISTRY[optimizer_name]
        except ImportError:
            raise ValueError(
                f"Requested optimizer {optimizer_name} is not available since NVIDIA/apex is not installed"
            )

    try:
        import torch_optimizer as to

        return to.get(optimizer_name)
    except ImportError:
        raise ValueError(
            f"Requested optimizer {optimizer_name} is not available since torch-optimizer is not installed. "
            f"Please install it from https://github.com/jettify/pytorch-optimizer"
        )
    except ValueError:
        raise


def scale_learning_rate_for_ddp(optimizer_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scale learning rate with respect to world size for Distributed Data Parallel training.
    Efficient learning rate is WORLD_SIZE * learning rate from config
    For non-distributed runs WORLD_SIZE is 1
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        optimizer_params = copy.deepcopy(optimizer_params)
        optimizer_params["lr"] = optimizer_params["lr"] * torch.distributed.get_world_size()

    return optimizer_params


def get_optimizer(
    model: nn.Module, optimizer_name: str, optimizer_params: Dict[str, Any], apply_weight_decay_to_bias: bool = True,
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
        parameters, **optimizer_params,
    )

    if not apply_weight_decay_to_bias:
        optimizer.add_param_group({"params": biases_pg, "weight_decay": 0.0})

    return optimizer
