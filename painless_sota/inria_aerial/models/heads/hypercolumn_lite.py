__all__ = ["FusePredictionsHead"]

from typing import List, Optional
import torch.nn.functional as F
import torch
from torch import nn


class FusePredictionsHead(nn.Module):
    def __init__(self, feature_maps: List[int], num_classes:int,output_name: Optional[str] = None):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(in_channels, num_classes, kernel_size=1) for in_channels in feature_maps]
        )
        self.output_name=output_name

    def forward(self, feature_maps, output_size):
        masks = []
        for projection, map in zip(self.projections, feature_maps):
            mask = projection(map)
            mask = F.interpolate(mask, size=output_size, mode="nearest")
            masks.append(mask)

        output = torch.stack(masks, dim=0).sum(dim=0)
        
        if self.output_name is not None:
            return {self.output_name: output}
        else:
            return output
