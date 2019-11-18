from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DecoderModule
from ..activated_batch_norm import ABN
from ..encoders import EncoderModule

__all__ = ["DeeplabV3Decoder"]


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes: int, output_stride: int, output_features: int, dropout=0.5):
        super(ASPP, self).__init__()

        if output_stride == 32:
            dilations = [1, 3, 6, 9]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, output_features, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, output_features, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, output_features, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, output_features, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, output_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(1280, output_features, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




class DeeplabV3Decoder(DecoderModule):
    def __init__(self, feature_maps: List[int], num_classes: int, dropout=0.5):
        super(DeeplabV3Decoder, self).__init__()

        low_level_features = feature_maps[0]
        high_level_features = feature_maps[-1]

        self.conv1 = nn.Conv2d(low_level_features, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.last_conv = nn.Sequential(
            nn.Conv2d(high_level_features + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.2),  # 5 times smaller dropout rate
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        self.reset_parameters()

    def forward(self, feature_maps):
        high_level_features = feature_maps[-1]
        low_level_feat = feature_maps[0]

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        high_level_features = F.interpolate(
            high_level_features, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True
        )
        high_level_features = torch.cat((high_level_features, low_level_feat), dim=1)
        high_level_features = self.last_conv(high_level_features)

        return high_level_features

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
