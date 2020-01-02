import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from .activations import (
    ACT_LEAKY_RELU,
    ACT_HARD_SWISH,
    ACT_MISH,
    ACT_SWISH,
    ACT_SELU,
    ACT_ELU,
    ACT_RELU6,
    ACT_RELU,
    ACT_HARD_SIGMOID,
    ACT_NONE,
    hard_sigmoid,
    hard_swish,
    mish,
    swish,
    ACT_SWISH_NAIVE, swish_naive)

__all__ = ["ABN"]


class ABN(nn.Module):
    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
        "num_features",
        "affine",
    ]

    """Activated Batch Normalization
    This gathers a `BatchNorm` and an activation function in a single module
    """

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        activation="leaky_relu",
        slope=0.01,
    ):
        """Create an Activated Batch Normalization module
        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

        self.activation = activation
        self.slope = slope

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        x = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

        if self.activation == ACT_RELU:
            return F.relu(x, inplace=True)
        elif self.activation == ACT_RELU6:
            return F.relu6(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return F.elu(x, inplace=True)
        elif self.activation == ACT_SELU:
            return F.selu(x, inplace=True)
        elif self.activation == ACT_SWISH:
            return swish(x)
        elif self.activation == ACT_SWISH_NAIVE:
            return swish_naive(x)
        elif self.activation == ACT_MISH:
            return mish(x)
        elif self.activation == ACT_HARD_SWISH:
            return hard_swish(x, inplace=True)
        elif self.activation == ACT_HARD_SIGMOID:
            return hard_sigmoid(x, inplace=True)
        elif self.activation == ACT_NONE:
            return x
        else:
            raise KeyError(self.activation)

    def __repr__(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}, activation={activation}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
