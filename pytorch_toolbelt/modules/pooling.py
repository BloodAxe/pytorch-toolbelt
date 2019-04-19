"""Implementation of different pooling modules

"""
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


class GWAP(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=True)
        return x


class RMSPool(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):
        x_mean = x.mean(dim=[2, 3])
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return avg_pool.sqrt()


class MILCustomPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weight_generator = nn.Sequential(nn.BatchNorm2d(in_channels),
                                              nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1),
                                              nn.Sigmoid())

    def forward(self, x):
        weight = self.weight_generator(x)
        loss = self.classifier(x)
        logits = torch.sum(weight * loss, dim=[2, 3]) / (torch.sum(weight, dim=[2, 3]) + 1e-6)
        return logits
