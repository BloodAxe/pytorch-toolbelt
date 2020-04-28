from torch import nn, Tensor

__all__ = ["DecoderModule", "SegmentationDecoderModule"]

from typing import List


class DecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:  # skipcq: PYL-W0221
        raise NotImplementedError

    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)


class SegmentationDecoderModule(DecoderModule):
    """
    A placeholder for future. Indicates sub-class decoders are suitable for segmentation tasks
    """

    pass
