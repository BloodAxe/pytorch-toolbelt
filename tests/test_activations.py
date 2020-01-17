import torch
import pytest

from pytorch_toolbelt.modules.activations import instantiate_activation_block

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


@pytest.mark.parametrize(
    "activation_name",
    ["none", "relu", "relu6", "leaky_relu", "elu", "selu", "celu", "mish", "swish", "hard_sigmoid", "hard_swish"],
)
def test_activations(activation_name):
    act = instantiate_activation_block(activation_name)
    x = torch.randn(128).float()
    y = act(x)
    assert y.dtype == torch.float32


@pytest.mark.parametrize(
    "activation_name",
    ["none", "relu", "relu6", "leaky_relu", "elu", "selu", "celu", "mish", "swish", "hard_sigmoid", "hard_swish"],
)
@skip_if_no_cuda
def test_activations_cuda(activation_name):
    act = instantiate_activation_block(activation_name)
    x = torch.randn(128).float().cuda()
    y = act(x)
    assert y.dtype == torch.float32

    act = instantiate_activation_block(activation_name)
    x = torch.randn(128).half().cuda()
    y = act(x)
    assert y.dtype == torch.float16
