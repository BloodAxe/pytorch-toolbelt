import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeeplabV3Decoder"]


class DeeplabV3Decoder(nn.Module):
    def __init__(
        self,
        high_level_features: int,
        low_level_features: int,
        num_classes: int,
        dropout=0.5,
    ):
        super(DeeplabV3Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_features, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                high_level_features + 48,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
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

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(
            x, size=low_level_feat.size()[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
