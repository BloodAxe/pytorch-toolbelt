from typing import List

import torch
import torch.nn.functional as F
from torch import nn


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """
    3x3 convolution + BN + relu
    """
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class UPerNet(nn.Module):
    def __init__(self, output_filters: List[int], num_classes=150, pool_scales=(1, 2, 3, 6), fpn_dim=256):
        super(UPerNet, self).__init__()

        last_fm_dim = output_filters[-1]

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(last_fm_dim, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(last_fm_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in output_filters[:-1]:  # skip the top layer
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(output_filters) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(output_filters) * fpn_dim, fpn_dim, 1), nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, feature_maps):
        last_fm = feature_maps[-1]

        input_size = last_fm.size()
        ppm_out = [last_fm]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(
                pool_conv(
                    F.interpolate(
                        pool_scale(last_fm), (input_size[2], input_size[3]), mode="bilinear", align_corners=False
                    )
                )
            )
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(feature_maps) - 1)):
            conv_x = feature_maps[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(f, size=conv_x.size()[2:], mode="bilinear", align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(fpn_feature_list[i], output_size, mode="bilinear", align_corners=False))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        return x
