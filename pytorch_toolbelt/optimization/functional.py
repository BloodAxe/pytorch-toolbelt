from typing import Optional, Iterator, Dict
from torch import nn

__all__ = ["get_lr_decay_parameters", "get_optimizable_parameters", "freeze_model"]


def get_lr_decay_parameters(model: nn.Module, learning_rate: float, lr_multipliers: Dict[str, float]):
    """
    Create different parameter groups with different settings
    Args:
        parameters:
        learning_rate:
        groups: {"encoder": 0.1 ,"encoder.layer2": 0.2}

    Returns:

    """
    custom_lr_parameters = dict(
        (group_name, {"params": [], "lr": learning_rate * lr_factor})
        for (group_name, lr_factor) in lr_multipliers.items()
    )
    custom_lr_parameters["default"] = {"params": [], "lr": learning_rate}

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        matches = False
        for group_name, lr in lr_multipliers.items():
            if str.startswith(parameter_name, group_name):
                custom_lr_parameters[group_name]["params"].append(parameter)
                matches = True
                break

        if not matches:
            custom_lr_parameters["default"]["params"].append(parameter)

    return list(custom_lr_parameters.values())


def get_optimizable_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """
    Return list of parameters with requires_grad=True from the model.
    This function allows easily get all parameters that should be optimized.
    :param model: An instance of nn.Module.
    :return: Parameters with requires_grad=True.
    """
    return filter(lambda x: x.requires_grad, model.parameters())


def freeze_model(module: nn.Module, freeze_parameters: Optional[bool] = True, freeze_bn: Optional[bool] = True):
    """
    Change 'requires_grad' value for module and it's child modules and
    optionally freeze batchnorm modules.
    :param module: Module to change
    :param freeze_parameters: True to freeze parameters; False - to enable parameters optimization.
        If None - current state is not changed.
    :param freeze_bn: True to freeze batch norm; False - to enable BatchNorm updates.
        If None - current state is not changed.
    :return: None
    """
    bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm

    if freeze_parameters is not None:
        for param in module.parameters():
            param.requires_grad = not freeze_parameters

    if freeze_bn is not None:
        if isinstance(module, bn_types):
            module.track_running_stats = not freeze_bn

        for m in module.modules():
            if isinstance(m, bn_types):
                module.track_running_stats = not freeze_bn
