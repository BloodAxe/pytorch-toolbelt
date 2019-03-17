import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return F.avg_pool2d(inputs, kernel_size=in_size[2:])


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return F.max_pool2d(inputs, kernel_size=in_size[2:])


