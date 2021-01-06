from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam, RMSprop, AdamW

__all__ = ["get_optimizer"]


def get_optimizer(
    optimizer_name: str, parameters, learning_rate: float, weight_decay=0, eps=1e-5, **kwargs
) -> Optimizer:
    """

    Args:
        optimizer_name:
        parameters:
        learning_rate:
        weight_decay:
        eps: Default eps value is 1e-5 to prevent gradient explosion during long training sessions in Adam-like optimizers
        **kwargs:

    Returns:

    """
    if optimizer_name.lower() == "sgd":
        return SGD(parameters, learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "adam":
        return Adam(parameters, learning_rate, weight_decay=weight_decay, eps=eps, **kwargs)

    if optimizer_name.lower() == "rms":
        return RMSprop(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "adamw":
        return AdamW(parameters, learning_rate, weight_decay=weight_decay, eps=eps, **kwargs)

    if optimizer_name.lower() == "radam":
        from torch_optimizer import RAdam

        return RAdam(parameters, learning_rate, weight_decay=weight_decay, eps=eps, **kwargs)  # As Jeremy suggests

    # Optimizers from torch-optimizer
    if optimizer_name.lower() == "ranger":
        from torch_optimizer import Ranger

        return Ranger(parameters, learning_rate, eps=eps, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "lamb":
        from torch_optimizer import Lamb

        return Lamb(parameters, learning_rate, eps=eps, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "diffgrad":
        from torch_optimizer import DiffGrad

        return DiffGrad(parameters, learning_rate, eps=eps, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "novograd":
        from torch_optimizer import NovoGrad

        return NovoGrad(parameters, learning_rate, eps=eps, weight_decay=weight_decay, **kwargs)

    # Optimizers from Apex (Fused version is faster on GPU with tensor cores)
    if optimizer_name.lower() == "fused_lamb":
        from apex.optimizers import FusedLAMB

        return FusedLAMB(parameters, learning_rate, eps=eps, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "fused_sgd":
        from apex.optimizers import FusedSGD

        return FusedSGD(parameters, learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == "fused_adam":
        from apex.optimizers import FusedAdam

        return FusedAdam(parameters, learning_rate, eps=eps, weight_decay=weight_decay, adam_w_mode=True, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)
