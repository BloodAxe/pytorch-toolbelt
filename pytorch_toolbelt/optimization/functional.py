import collections
from typing import Optional, Iterator, Dict, Union, List
from torch import nn

__all__ = ["get_lr_decay_parameters", "get_optimizable_parameters", "freeze_model"]


def build_optimizer_param_groups(
    model: nn.Module,
    
    learning_rate: Union[float, Dict[str, float]],
    weight_decay: Union[float, Dict[str, float]],
    apply_weight_decay_on_bias: bool,
    apply_weight_decay_on_norm: bool,
    layerwise_learning_rate: Dict[str, float],
    layerwise_weight_decay: Dict[str,float]
) -> collections.OrderedDict:
    """

    Args:
        model:
        learning_rate:
        weight_decay:
        apply_weight_decay_on_bias: If True, weight decay is applied to bias on Linear & Conv layers
        apply_weight_decay_on_norm: If True, weight decay is applied normalization layers

    Returns:

    """
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


def test_optimizer_groups():
    def conv_bn_relu(in_channels, out_channels):
        return nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

    model = nn.Sequential(
        collections.OrderedDict(
            [
                ("encoder", nn.Sequential(conv_bn_relu(3, 32), conv_bn_relu(32, 64))),
                ("neck", nn.Sequential(conv_bn_relu(64, 64), conv_bn_relu(64, 64))),
                ("decoder", nn.Sequential(conv_bn_relu(64, 32), conv_bn_relu(32, 1))),
            ]
        )
    )

    pg = build_optimizer_param_groups(
        model, learning_rate=1e-4, weight_decay=0, apply_weight_decay_on_bias=False, apply_weight_decay_on_norm=False
    )

    pg = build_optimizer_param_groups(
        model,
        learning_rate=1e-2,
        weight_decay=0,
        layerwise_learning_rate={"encoder": 1e-3, "neck": 1e-4, "decoder": 1e-5},
        layerwise_weight_decay={"encoder.0": 1e-5, "encoder.1": 1e-55, },
        apply_weight_decay_on_bias=False,
        apply_weight_decay_on_norm=False,
    )

    assert len(pg) == 1
