"""Implementation of different pooling modules

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GlobalAvgPool2d",
    "GlobalMaxPool2d",
    "GlobalWeightedAvgPool2d",
    "GWAP",
    "RMSPool",
    "MILCustomPoolingModule",
    "GlobalRankPooling",
]


class GlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):  # skipcq: PYL-W0221
        x = F.adaptive_avg_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalMaxPool2d(nn.Module):
    def __init__(self, flatten=False):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalMaxPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):  # skipcq: PYL-W0221
        x = F.adaptive_max_pool2d(x, output_size=1)
        if self.flatten:
            x = x.view(x.size(0), x.size(1))
        return x


class GlobalWeightedAvgPool2d(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average
    Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features: int, flatten=False):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.flatten = flatten

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):  # skipcq: PYL-W0221
        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3], keepdim=not self.flatten)
        return x


GWAP = GlobalWeightedAvgPool2d


class RMSPool(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):  # skipcq: PYL-W0221
        x_mean = x.mean(dim=[2, 3])
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return avg_pool.sqrt()


class MILCustomPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weight_generator = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # skipcq: PYL-W0221
        weight = self.weight_generator(x)
        loss = self.classifier(x)
        logits = torch.sum(weight * loss, dim=[2, 3]) / (torch.sum(weight, dim=[2, 3]) + 1e-6)
        return logits


class GlobalRankPooling(nn.Module):
    """
    https://arxiv.org/abs/1704.02112
    """

    def __init__(self, num_features, spatial_size, flatten=False):
        super().__init__()
        self.conv = nn.Conv1d(num_features, num_features, spatial_size, groups=num_features)
        self.flatten = flatten

    def forward(self, x: torch.Tensor):  # skipcq: PYL-W0221
        spatial_size = x.size(2) * x.size(3)
        assert spatial_size == self.conv.kernel_size[0], (
            f"Expected spatial size {self.conv.kernel_size[0]}, " f"got {x.size(2)}x{x.size(3)}"
        )

        x = x.view(x.size(0), x.size(1), -1)  # Flatten spatial dimensions
        x_sorted, index = x.topk(spatial_size, dim=2)

        x = self.conv(x_sorted)  # [B, C, 1]

        if self.flatten:
            x = x.squeeze(2)
        return x
