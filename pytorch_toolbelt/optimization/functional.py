import collections
from typing import Optional, Iterator, Dict, Union, List
from torch import nn
import itertools

__all__ = ["get_lr_decay_parameters", "get_optimizable_parameters", "freeze_model", "build_optimizer_param_groups"]


def build_optimizer_param_groups(
    model: nn.Module,
    learning_rate: Union[float, Dict[str, float]],
    weight_decay: Union[float, Dict[str, float]],
    apply_weight_decay_on_bias: bool = True,
    apply_weight_decay_on_norm: bool = True,
) -> collections.OrderedDict:
    """

    Args:
        model:
        learning_rate: A single number of dictionary of layerwise learning rate parameters.
        weight_decay: A single number of dictionary of layerwise weight decay parameters.
        apply_weight_decay_on_bias: If True, weight decay is applied to bias on Linear & Conv layers (default).
            This parameter is False, it overrule the matching layerwise weight-decay parameter.
        apply_weight_decay_on_norm: If True, weight decay is applied normalization layers (default).
            This parameter is False, it overrule the matching layerwise weight-decay parameter.

    Returns:

    """
    if isinstance(learning_rate) and "_default_" not in learning_rate:
        raise RuntimeError("When using layerwise learning rate, a key _default_ must be present to indicate default LR")

    if isinstance(weight_decay) and "_default_" not in weight_decay:
        raise RuntimeError("When using layerwise weight decay, a key _default_ must be present to indicate default LR")

    all_params:List[Tuple[str,nn.Module]] = list(model.named_modules())
    layerwise_groups = itertools.product(
        [k for k in learning_rate.keys() if k != "_default_"],
        [k for k in weight_decay.keys() if k != "_default"]
    )

    for lr_prefix, wd_prefix in layerwise_groups:
        remaining_params = []
        for module_name, module in all_params:
            if module_name.startswith(lr_prefix) and module_name.startswith(wd_prefix)
            else:
                remaining_params.append((module_name, module))

        all_params = remaining_params


    parameter_groups = collections.OrderedDict()
    default_pg = []
    no_wd_on_bias_pg = []
    no_wd_on_norm_pg = []

    for module_name, module in model.named_modules():
        if isinstance(module, (nn._BatchNorm, nn._InstanceNorm)) and not apply_weight_decay_on_norm:
            no_wd_on_bias_pg.append((module_name, get_optimizable_parameters(module)))
        elif isinstance(module, (nn.Linear, nn._ConvNd, nn._ConvTransposeNd)) and not apply_weight_decay_on_bias:
            no_wd_on_norm_pg.append()
        else:
            default_pg.append((module_name, get_optimizable_parameters(module)))

    if len(default_pg):
        parameter_groups["default"] = default_pg

    if len(no_wd_on_bias_pg):
        parameter_groups["no_wd_on_bias_pg"] = no_wd_on_bias_pg

    if len(no_wd_on_norm_pg):
        parameter_groups["no_wd_on_norm_pg"] = no_wd_on_norm_pg

    return default_pg


def get_lr_decay_parameters(model: nn.Module, learning_rate: float, lr_multipliers: Dict[str, float]):
    """
    Create different parameter groups with different settings.

    Args:
        parameters:
        learning_rate:
        groups: {"encoder": 0.1 ,"encoder.layer2": 0.2}
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


def freeze_model(
    module: nn.Module, freeze_parameters: Optional[bool] = True, freeze_bn: Optional[bool] = True
) -> nn.Module:
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

    return module
