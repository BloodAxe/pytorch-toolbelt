from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .identity import Identity

__all__ = [
    "ABN",
    "ACT_ELU",
    "ACT_GELU",
    "ACT_HARD_SIGMOID",
    "ACT_HARD_SWISH",
    "ACT_LEAKY_RELU",
    "ACT_MISH",
    "ACT_MISH_NAIVE",
    "ACT_NONE",
    "ACT_RELU",
    "ACT_RELU6",
    "ACT_SELU",
    "ACT_SIGMOID",
    "ACT_SILU",
    "ACT_SOFTPLUS",
    "ACT_SWISH",
    "ACT_SWISH_NAIVE",
    "AGN",
    "HardSigmoid",
    "HardSwish",
    "Mish",
    "MishNaive",
    "Swish",
    "SwishNaive",
    "get_activation_block",
    "hard_sigmoid",
    "hard_swish",
    "instantiate_activation_block",
    "mish",
    "mish_naive",
    "sanitize_activation_name",
    "swish",
    "swish_naive",
]

# Activation names
ACT_CELU = "celu"
ACT_ELU = "elu"
ACT_GELU = "gelu"
ACT_GLU = "glu"
ACT_HARD_SIGMOID = "hard_sigmoid"
ACT_HARD_SWISH = "hard_swish"
ACT_LEAKY_RELU = "leaky_relu"
ACT_MISH = "mish"
ACT_MISH_NAIVE = "mish_naive"
ACT_NONE = "none"
ACT_PRELU = "prelu"
ACT_RELU = "relu"
ACT_RELU6 = "relu6"
ACT_SELU = "selu"
ACT_SIGMOID = "sigmoid"
ACT_SILU = "silu"
ACT_SOFTPLUS = "softplus"
ACT_SWISH = "swish"
ACT_SWISH_NAIVE = "swish_naive"

# This version reduces memory overhead of Swish during training by
# recomputing torch.sigmoid(x) in backward instead of saving it.
@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishFunction(torch.autograd.Function):
    """
    Memory efficient Swish implementation.

    Credit:
        https://blog.ceshine.net/post/pytorch-memory-swish/
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/activations_jit.py

    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish(x):
    return SwishFunction.apply(x)


def swish_naive(x):
    return x * x.sigmoid()


def mish_naive(input):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return input * torch.tanh(F.softplus(input))


class MishNaive(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()

    def forward(self, x):
        return mish_naive(x)


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishFunction(torch.autograd.Function):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish(x):
    """
    Apply the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    Credit: https://github.com/digantamisra98/Mish
    """
    return MishFunction.apply(x)


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

    def __init__(self, inplace=False):
        """
        Init method.
        :param inplace: Not used, exists only for compatibility
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)


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
    def __init__(self, inplace=False):
        super(Swish, self).__init__()

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
        ACT_ELU: nn.ELU,
        ACT_GELU: nn.GELU,
        ACT_GLU: nn.GLU,
        ACT_HARD_SIGMOID: HardSigmoid,
        ACT_HARD_SWISH: HardSwish,
        ACT_LEAKY_RELU: nn.LeakyReLU,
        ACT_MISH: Mish,
        ACT_MISH_NAIVE: MishNaive,
        ACT_NONE: Identity,
        ACT_PRELU: nn.PReLU,
        ACT_RELU6: nn.ReLU6,
        ACT_RELU: nn.ReLU,
        ACT_SELU: nn.SELU,
        ACT_SILU: nn.SiLU,
        ACT_SOFTPLUS: nn.Softplus,
        ACT_SWISH: Swish,
        ACT_SWISH_NAIVE: SwishNaive,
        ACT_SIGMOID: nn.Sigmoid
    }

    return ACTIVATIONS[activation_name.lower()]


def instantiate_activation_block(activation_name: str, **kwargs) -> nn.Module:
    block = get_activation_block(activation_name)

    act_params = {}

    if "inplace" in kwargs and activation_name in {ACT_RELU, ACT_RELU6, ACT_LEAKY_RELU, ACT_SELU, ACT_CELU, ACT_ELU}:
        act_params["inplace"] = kwargs["inplace"]

    if "slope" in kwargs and activation_name in {ACT_LEAKY_RELU}:
        act_params["negative_slope"] = kwargs["slope"]

    return block(**act_params)


def sanitize_activation_name(activation_name: str) -> str:
    """
    Return reasonable activation name for initialization in `kaiming_uniform_` for hipster activations
    """
    if activation_name in {ACT_MISH, ACT_SWISH, ACT_SWISH_NAIVE, ACT_MISH_NAIVE}:
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
