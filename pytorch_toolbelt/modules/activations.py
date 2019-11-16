from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from .identity import Identity

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
    "ACT_MISH",
    "mish",
    "swish",
    "hard_sigmoid",
    "hard_swish",
    "HardSigmoid",
    "HardSwish",
    "Swish",
    "get_activation_module",
    "sanitize_activation_name",
]

# Activation names
ACT_RELU = "relu"
ACT_RELU6 = "relu6"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"
ACT_SELU = "selu"
ACT_SWISH = "swish"
ACT_MISH = "mish"
ACT_HARD_SWISH = "hard_swish"
ACT_HARD_SIGMOID = "hard_sigmoid"


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.

    Credit: https://blog.ceshine.net/post/pytorch-memory-swish/
    """

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Credit: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)


def swish(x):
    return SwishFunction.apply(x)


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
    def forward(self, input_tensor):
        return SwishFunction.apply(input_tensor)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def get_activation_module(activation_name: str, **kwargs) -> nn.Module:
    ACTIVATIONS = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "glu": nn.GLU,
        "prelu": nn.PReLU,
        "swish": Swish,
        "mish": Mish,
        "hard_sigmoid": HardSigmoid,
        "hard_swish": HardSwish,
        "none": Identity,
    }

    return ACTIVATIONS[activation_name.lower()](**kwargs)


def sanitize_activation_name(activation_name: str) -> str:
    """
    Return reasonable activation name for initialization in `kaiming_uniform_` for hipster activations
    """
    if activation_name in {ACT_MISH, ACT_SWISH}:
        return ACT_LEAKY_RELU

    return activation_name
