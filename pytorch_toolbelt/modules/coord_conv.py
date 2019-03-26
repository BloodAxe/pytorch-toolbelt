"""Implementation of the CoordConv modules from https://arxiv.org/abs/1807.03247

"""

import torch
import torch.nn as nn


def append_coords(input_tensor, with_r=False):
    batch_size, _, x_dim, y_dim = input_tensor.size()

    xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

    ret = torch.cat([
        input_tensor,
        xx_channel.type_as(input_tensor),
        yy_channel.type_as(input_tensor)], dim=1)

    if with_r:
        rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
        ret = torch.cat([ret, rr], dim=1)

    return ret


'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
'''


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        return append_coords(input_tensor, self.with_r)


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
