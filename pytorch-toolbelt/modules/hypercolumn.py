"""Implementation of hypercolumn module from "Hypercolumns for Object Segmentation and Fine-grained Localization"

Original paper: https://arxiv.org/abs/1411.5752
"""

import torch
from torch import nn
from torch.nn import functional as F


class HyperColumn(nn.Module):
    def __init__(self, mode='bilinear', align_corners=True):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, *features):
        layers = []
        dst_size = features[0].size()[-2:]

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)
