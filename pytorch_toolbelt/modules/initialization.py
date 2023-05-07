from typing import Union

import torch.nn.init
from torch import nn

__all__ = ["first_class_background_init"]


def first_class_background_init(
    module: Union[nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d], background_prob: float = 0.95
):
    """
    Initialize weights of the Linear or Conv layer with zeros and bias with values of
    [logit(bg_prob), logit(fg_prob), logit(fg_prob),  ...]
    """
    bg_bias = torch.logit(torch.tensor(background_prob))
    fg_bias = torch.logit(torch.tensor((1 - background_prob)))

    torch.nn.init.zeros_(module.weight)
    torch.nn.init.constant_(module.bias, bg_bias)
    torch.nn.init.constant_(module.bias[1:], fg_bias)
