import dataclasses
import numbers
from typing import Optional, Iterator, Dict, Union, List, Tuple, Mapping

from torch import nn

__all__ = ["get_lr_decay_parameters", "get_optimizable_parameters", "freeze_model", "build_optimizer_param_groups"]


@dataclasses.dataclass
class ParametersGroup:
    lr: Union[float, None]
    weight_decay: Union[float, None]
    params: List
    name: str

    def __len__(self):
        return len(self.params)

    def asdict(self):
        d = {"params": self.params, "name": self.name}
        if self.lr is not None:
            d["lr"] = self.lr
        if self.weight_decay is not None:
            d["weight_decay"] = self.weight_decay
        return d


def rec_getattr(obj, attr):
    """Get object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_getattr(a, 'b.c')
    4
    """
    if "." not in attr:
        return getattr(obj, attr)
    else:
        L = attr.split(".")
        return rec_getattr(getattr(obj, L[0]), ".".join(L[1:]))


def build_optimizer_param_groups(
    model: nn.Module,
    learning_rate: Union[float, Mapping[str, float]],
    weight_decay: Union[float, Mapping[str, float]],
    apply_weight_decay_on_bias: bool = True,
    apply_weight_decay_on_norm: bool = True,
) -> Tuple[List[ParametersGroup], Dict]:
    """

    Args:
        model:
        learning_rate: A single number of dictionary of layer-wise learning rate parameters.
        weight_decay: A single number of dictionary of layer-wise weight decay parameters.
        apply_weight_decay_on_bias: If True, weight decay is applied to bias on Linear & Conv layers (default).
            This parameter is False, it overrules the matching layer-wise weight-decay parameter.
        apply_weight_decay_on_norm: If True, weight decay is applied normalization layers (default).
            This parameter is False, it overrules the matching layer-wise weight-decay parameter.

    Returns:

    """
    if isinstance(learning_rate, Mapping) and "_default_" not in learning_rate:
        raise RuntimeError(
            "When using layerwise learning rate, a key _default_ must be present to indicate default LR"
        )

    if isinstance(weight_decay, Mapping) and "_default_" not in weight_decay:
        raise RuntimeError("When using layerwise weight decay, a key _default_ must be present to indicate default LR")

    if isinstance(learning_rate, numbers.Number):
        learning_rate = {"_default_": float(learning_rate)}
    if isinstance(weight_decay, numbers.Number):
        weight_decay = {"_default_": float(weight_decay)}

    default_learning_rate = float(learning_rate["_default_"])
    default_weight_decay = float(weight_decay["_default_"])

    learning_rate = list(learning_rate.items())
    weight_decay = list(weight_decay.items())

    parameter_groups = {}

    norm_layers = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm3d,
        nn.InstanceNorm2d,
    )

    layers_with_bias = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )

    def get_param_group(parameter_name: str, module: nn.Module):
        matching_lr_index = -1
        matching_lr_value = default_learning_rate
        for lr_index, (prefix, lr) in enumerate(learning_rate):
            if parameter_name.startswith(prefix):
                matching_lr_index = lr_index
                matching_lr_value = lr
                break

        matching_wd_index = -1
        matching_wd_value = default_weight_decay
        for wd_index, (prefix, wd) in enumerate(weight_decay):
            if parameter_name.startswith(prefix):
                matching_wd_index = wd_index
                matching_wd_value = wd
                break

        if not apply_weight_decay_on_norm and isinstance(module, norm_layers):
            matching_wd_value = 0
            matching_wd_index = "_no_wd_on_norm"

        if (
            not apply_weight_decay_on_bias
            and isinstance(module, layers_with_bias)
            and parameter_name.endswith(".bias")
        ):
            matching_wd_value = 0
            matching_wd_index = "_no_wd_on_bias"

        param_group_name = f"{matching_lr_index}_{matching_wd_index}"
        if param_group_name not in parameter_groups:
            parameter_groups[param_group_name] = ParametersGroup(
                lr=matching_lr_value, weight_decay=matching_wd_value, name=param_group_name, params=[]
            )

        return parameter_groups[param_group_name]

    # All parameters
    parameters = get_named_optimizable_parameters(model)

    for parameter_name, parameter in parameters:
        module_name = ".".join(parameter_name.split(".")[:-1])
        module = rec_getattr(model, module_name)

        param_group: ParametersGroup = get_param_group(parameter_name, module)
        param_group.params.append(parameter)
    
    defaults = {"lr": default_learning_rate, "weight_decay": default_weight_decay}
    pgs = [x.asdict() for x in parameter_groups.values()]
    return pgs, defaults


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


def get_named_optimizable_parameters(model: nn.Module, prefix: str = "") -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Return list of parameters with requires_grad=True from the model.
    This function allows easily get all parameters that should be optimized.
    :param model: An instance of nn.Module.
    :return: Parameters with requires_grad=True.
    """
    return filter(lambda x: x[1].requires_grad, model.named_parameters(prefix))


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
