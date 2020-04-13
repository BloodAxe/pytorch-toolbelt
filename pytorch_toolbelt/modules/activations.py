from collections import OrderedDict

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
    "instantiate_activation_block",
    "get_activation_block",
    "sanitize_activation_name",
    "ABN",
    "AGN",
]

# Activation names
ACT_CELU = "celu"
ACT_ELU = "elu"
ACT_GLU = "glu"
ACT_HARD_SIGMOID = "hard_sigmoid"
ACT_HARD_SWISH = "hard_swish"
ACT_LEAKY_RELU = "leaky_relu"
ACT_MISH = "mish"
ACT_NONE = "none"
ACT_PRELU = "prelu"
ACT_RELU = "relu"
ACT_RELU6 = "relu6"
ACT_SELU = "selu"
ACT_SWISH = "swish"
ACT_SWISH_NAIVE = "swish_naive"


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
    Apply the mish function element-wise:
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


def swish_naive(x):
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


class SwishNaive(nn.Module):
    def forward(self, input_tensor):
        return swish_naive(input_tensor)


class Swish(nn.Module):
    def forward(self, input_tensor):
        return swish(input_tensor)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def get_activation_block(activation_name: str):
    ACTIVATIONS = {
        ACT_CELU: nn.CELU,
        ACT_GLU: nn.GLU,
        ACT_PRELU: nn.PReLU,
        ACT_ELU: nn.ELU,
        ACT_HARD_SIGMOID: HardSigmoid,
        ACT_HARD_SWISH: HardSwish,
        ACT_LEAKY_RELU: nn.LeakyReLU,
        ACT_MISH: Mish,
        ACT_NONE: Identity,
        ACT_RELU6: nn.ReLU6,
        ACT_RELU: nn.ReLU,
        ACT_SELU: nn.SELU,
        ACT_SWISH: Swish,
        ACT_SWISH_NAIVE: SwishNaive,
    }

    return ACTIVATIONS[activation_name.lower()]


def instantiate_activation_block(activation_name: str, **kwargs) -> nn.Module:
    block = get_activation_block(activation_name)

    act_params = {}

    if "inplace" in kwargs and activation_name in {ACT_RELU, ACT_RELU6, ACT_LEAKY_RELU, ACT_SELU, ACT_CELU, ACT_ELU}:
        act_params["inplace"] = kwargs["inplace"]

    if "slope" in kwargs and activation_name in {ACT_LEAKY_RELU}:
        act_params["slope"] = kwargs["slope"]

    return block(**act_params)


def sanitize_activation_name(activation_name: str) -> str:
    """
    Return reasonable activation name for initialization in `kaiming_uniform_` for hipster activations
    """
    if activation_name in {ACT_MISH, ACT_SWISH, ACT_SWISH_NAIVE}:
        return ACT_LEAKY_RELU

    return activation_name


def ABN(
    num_features: int,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    activation=ACT_RELU,
    slope=0.01,
    inplace=True,
):
    bn = nn.BatchNorm2d(
        num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
    )
    act = instantiate_activation_block(activation, inplace=inplace, slope=slope)
    return nn.Sequential(OrderedDict([("bn", bn), (activation, act)]))


def AGN(num_features: int, num_groups: int, eps=1e-5, affine=True, activation=ACT_RELU, slope=0.01, inplace=True):
    gn = nn.GroupNorm(num_channels=num_features, num_groups=num_groups, eps=eps, affine=affine)
    act = instantiate_activation_block(activation, inplace=inplace, slope=slope)
    return nn.Sequential(OrderedDict([("gn", gn), (activation, act)]))
