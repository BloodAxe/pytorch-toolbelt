from typing import Optional, Mapping, Union, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


__all__ = ["SimpleSegmentationHead"]


class SimpleSegmentationHead(nn.Module):
    def __init__(self, channels: List[int], num_classes: int, output_name: Optional[str] = None, dropout_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.drop = nn.Dropout2d(dropout_rate)
        self.final = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.output_name = output_name

        torch.nn.init.constant_(self.final.bias, -4)
        if num_classes > 1:
            torch.nn.init.constant_(self.final.bias, -4)
            torch.nn.init.constant_(self.final.bias[0], 4)
        else:
            torch.nn.init.constant_(self.final.bias[0], -1)

    def forward(self, features: List[Tensor], output_size: torch.Size) -> Union[Tensor, Mapping[str, Tensor]]:
        x = self.drop(features[0])
        x = self.final(x)
        output = F.interpolate(x, size=output_size, mode="bilinear", align_corners=True)
        if self.output_name is not None:
            return {self.output_name: output}
        else:
            return output


if __name__ == "__main__":
    from pytorch_toolbelt.utils import describe_outputs

    net = SimpleSegmentationHead(channels=64, num_classes=10, output_name="MASK")
    output = net(torch.randn(1, 64, 128, 128), output_size=(512, 512))
    assert "MASK" in output
    print(describe_outputs(output))
