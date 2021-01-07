from torch.optim.optimizer import Optimizer

__all__ = ["get_optimizer"]


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 1e-5,
    no_weight_decay_on_bias: bool = False,
    eps: float = 1e-5,
    **kwargs,
) -> Optimizer:
    """
    Construct an Optimizer for given model
    Args:
        model: Model to optimize. Only parameters that require_grad will be used
        optimizer_name: Name of the optimizer. Case-insensitive
        learning_rate: Target learning rate (regardless of the scheduler)
        weight_decay: Target weight decay
        no_weight_decay_on_bias: Whether to disable weight decay on bias parameters
        eps: Default epsilon for Adam-like optimizers.
        **kwargs: Additional parameters for optimizer

    Returns:

    """
    from torch.optim import ASGD, SGD, Adam, RMSprop, AdamW
    from torch_optimizer import RAdam, Lamb, DiffGrad, NovoGrad, Ranger

    # Optimizer parameter groups
    default_pg, biases_pg = [], []

    for k, v in model.named_parameters():
        if v.requires_grad:
            if str.endswith(k, ".bias"):
                biases_pg.append(v)  # biases
            else:
                default_pg.append(v)  # all else

    if no_weight_decay_on_bias:
        parameters = default_pg
    else:
        parameters = default_pg + biases_pg

    optimizer: Optimizer = None

    if optimizer_name.lower() == "sgd":
        optimizer = SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "asgd":
        optimizer = ASGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "adam":
        optimizer = Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )
    elif optimizer_name.lower() == "rms":
        optimizer = RMSprop(parameters, learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )
    elif optimizer_name.lower() == "radam":
        optimizer = RAdam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )
    elif optimizer_name.lower() == "ranger":
        optimizer = Ranger(
            parameters,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "lamb":
        optimizer = Lamb(
            parameters,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "diffgrad":
        optimizer = DiffGrad(
            parameters,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "novograd":
        optimizer = NovoGrad(
            parameters,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "fused_lamb":
        from apex.optimizers import FusedLAMB

        optimizer = FusedLAMB(parameters, learning_rate, eps=eps, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "fused_sgd":
        from apex.optimizers import FusedSGD

        optimizer = FusedSGD(
            parameters, learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs
        )
    elif optimizer_name.lower() == "fused_adam":
        from apex.optimizers import FusedAdam

        optimizer = FusedAdam(
            parameters, learning_rate, eps=eps, weight_decay=weight_decay, adam_w_mode=True, **kwargs
        )
    else:
        raise KeyError(f"Cannot get optimizer by name {optimizer_name}")

    # Currently either no_wd or per-group lr
    if no_weight_decay_on_bias:
        optimizer.add_param_group({"params": biases_pg, "weight_decay": 0})

    return optimizer
