import torch
from torch import nn

from pytorch_toolbelt.modules.initialization import first_class_background_init


def test_first_class_background_init_conv():
    conv = nn.Conv2d(256, 19, kernel_size=3, padding=1)
    first_class_background_init(conv, background_prob=0.96)

    input = torch.randn(4, 256, 64, 64)
    output = conv(input).softmax(dim=1)

    actual_probas = output.mean(dim=(0, 2, 3))
    assert actual_probas[0] > 0.96
    assert (actual_probas[1:] < 0.04).all()


def test_first_class_background_init_linear():
    conv = nn.Linear(256, 19)
    first_class_background_init(conv, background_prob=0.96)

    input = torch.randn(4, 256)
    output = conv(input).softmax(dim=1)

    actual_probas = output.mean(dim=(0))
    assert actual_probas[0] > 0.96
    assert (actual_probas[1:] < 0.04).all()
