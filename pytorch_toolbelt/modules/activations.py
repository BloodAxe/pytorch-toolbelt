from functools import partial

from torch import nn
from torch.nn import functional as F

__all__ = [
    "ACT_ELU",
    "ACT_HARD_SIGMOID",
    "ACT_HARD_SWISH",
    "ACT_LEAKY_RELU",
    "ACT_NONE",
    "ACT_RELU",
    "ACT_RELU6",
    "ACT_SELU",
    "ACT_SWISH",
    "swish",
    "hard_sigmoid",
    "hard_swish",
    "HardSigmoid",
    "HardSwish",
    "Swish",
    "get_activation_module",
    "sanitize_activation_name"
]

# Activation names
ACT_RELU = "relu"
ACT_RELU6 = "relu6"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"
ACT_SELU = "selu"
ACT_SWISH = "swish"
ACT_HARD_SWISH = "hard_swish"
ACT_HARD_SIGMOID = "hard_sigmoid"


def swish(x):
    return x * x.sigmoid()


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def get_activation_module(activation_name: str, **kwargs) -> nn.Module:
    if activation_name.lower() == "relu":
        return partial(nn.ReLU, **kwargs)

    if activation_name.lower() == "relu6":
        return partial(nn.ReLU6, **kwargs)

    if activation_name.lower() == "leaky_relu":
        return partial(nn.LeakyReLU, **kwargs)

    if activation_name.lower() == "elu":
        return partial(nn.ELU, **kwargs)

    if activation_name.lower() == "selu":
        return partial(nn.SELU, **kwargs)

    if activation_name.lower() == "celu":
        return partial(nn.CELU, **kwargs)

    if activation_name.lower() == "glu":
        return partial(nn.GLU, **kwargs)

    if activation_name.lower() == "prelu":
        return partial(nn.PReLU, **kwargs)

    if activation_name.lower() == "hard_sigmoid":
        return partial(HardSigmoid, **kwargs)

    if activation_name.lower() == "swish":
        return partial(Swish, **kwargs)

    if activation_name.lower() == "hard_swish":
        return partial(HardSwish, **kwargs)

    raise ValueError(f"Activation '{activation_name}' is not supported")


def sanitize_activation_name(activation_name):
    """
    Return reasonable activation name for initialization in `kaiming_uniform_` for hipster activations
    """
    if activation_name in {"swish", "mish"}:
        return "leaky_relu"

    return activation_name
