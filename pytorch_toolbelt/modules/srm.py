import torch
from torch import nn


class SRMLayer(nn.Module):
    """An implementation of SRM block, proposed in
    "SRM : A Style-based Recalibration Module for Convolutional Neural Networks".

    """

    def __init__(self, channels: int):
        super(SRMLayer, self).__init__()

        # Equal to torch.einsum('bck,ck->bc', A, B)
        self.cfc = nn.Conv1d(channels, channels, kernel_size=2, bias=False, groups=channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)
