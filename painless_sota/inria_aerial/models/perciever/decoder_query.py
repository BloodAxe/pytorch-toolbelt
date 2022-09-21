from abc import abstractmethod

from painless_sota.inria_aerial.models.perciever.position_encoding import PositionEncodingOutput
from torch import nn, Tensor

__all__ = ["DecoderQuery", "FourierPositionEncodingQuery", "EncoderInputQuery"]


class DecoderQuery(nn.Module):
    @abstractmethod
    def forward(self, x: PositionEncodingOutput, z: Tensor) -> Tensor:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_output_channels(self) -> int:
        raise NotImplementedError()


class FourierPositionEncodingQuery(DecoderQuery):
    def __init__(self, num_position_encoding_channels: int):
        super().__init__()
        self.num_position_encoding_channels = num_position_encoding_channels

    def forward(self, x: PositionEncodingOutput, z: Tensor) -> Tensor:
        return x.position_encoding

    @property
    @abstractmethod
    def num_output_channels(self) -> int:
        return self.num_position_encoding_channels


class EncoderInputQuery(DecoderQuery):
    def __init__(self, num_input_channels: int):
        super().__init__()
        self.num_input_channels = num_input_channels

    def forward(self, x: PositionEncodingOutput, z: Tensor) -> Tensor:
        return x.encoded_input

    @property
    @abstractmethod
    def num_output_channels(self) -> int:
        return self.num_input_channels
