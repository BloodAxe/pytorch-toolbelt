from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .common import DecoderModule


class PPMDecoder(DecoderModule):
    """
    Pyramid pooling decoder module

    https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/42b7567a43b1dab568e2bbfcbc8872778fbda92a/models/models.py
    """

    def __init__(self, feature_maps: List[int], num_classes=150, channels=512, pool_scales=(1, 2, 3, 6)):
        super(PPMDecoder, self).__init__()

        fc_dim = feature_maps[-1]
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )

    def forward(self, feature_maps: List[torch.Tensor]):
        last_fm = feature_maps[-1]

        input_size = last_fm.size()
        ppm_out = [last_fm]
        for pool_scale in self.ppm:
            input_pooled = pool_scale(last_fm)
            input_pooled = F.interpolate(input_pooled, size=input_size[2:], mode="bilinear", align_corners=False)
            ppm_out.append(input_pooled)
        ppm_out = torch.cat(ppm_out, dim=1)

        x = self.conv_last(ppm_out)
        return x
