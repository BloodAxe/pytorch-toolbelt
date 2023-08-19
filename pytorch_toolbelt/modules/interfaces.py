import dataclasses
from abc import abstractmethod
from typing import Protocol, Tuple, List, Union, Mapping, Callable

import numpy as np
import torch.jit

__all__ = [
    "FeatureMapsSpecification",
    "HasOutputFeaturesSpecification",
    "AbstractDecoder",
    "AbstractHead",
    "AbstractEncoder",
]

from torch import nn, Tensor

from pytorch_toolbelt.utils import pytorch_toolbelt_deprecated


@dataclasses.dataclass
class FeatureMapsSpecification:
    channels: Tuple[int, ...]
    strides: Tuple[int, ...]

    def __init__(self, channels: Union[Tuple[int, ...], List[int]], strides: Union[Tuple[int, ...], List[int]]):
        if len(channels) != len(strides):
            raise RuntimeError(
                f"Length of feature_map_channels({len(channels)} must"
                f" be equal to length of feature_map_strides({len(strides)})"
            )

        self.channels = tuple(channels)
        self.strides = tuple(strides)

    def get_index_of_largest_feature_map(self) -> int:
        """
        Returns index of largest (spatially) feature map (with smallest stride)
        :return: 0-based index of largest feature map
        """
        return int(np.argmin(self.strides))

    def get_dummy_input(self, device=None, image_size=(640, 512)) -> List[Tensor]:
        """
        Returns dummy input for this feature map specification
        :return: Dummy input tensor
        """
        feature_maps = []
        rows, cols = image_size
        for c, s in zip(self.channels, self.strides):
            feature_maps.append(torch.randn((1, c, rows // s, cols // s), device=device))
        return feature_maps

    def __len__(self) -> int:
        return len(self.channels)


class HasInputFeaturesSpecification(Protocol):
    """
    A protocol for modules that have output features
    """

    @torch.jit.unused
    def get_input_spec(self) -> FeatureMapsSpecification:
        ...


class HasOutputFeaturesSpecification(Protocol):
    """
    A protocol for modules that have output features
    """

    @torch.jit.unused
    def get_output_spec(self) -> FeatureMapsSpecification:
        ...


class AbstractEncoder(nn.Module, HasOutputFeaturesSpecification):
    pass


class AbstractDecoder(nn.Module, HasInputFeaturesSpecification, HasOutputFeaturesSpecification):
    __constants__ = ["input_spec"]

    def __init__(self, input_spec: FeatureMapsSpecification):
        if input_spec is None:
            raise ValueError("input_spec must be specified")
        super().__init__()
        self.input_spec = input_spec

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:  # skipcq: PYL-W0221
        raise NotImplementedError

    @torch.jit.unused
    def set_trainable(self, trainable):
        for param in self.parameters():
            param.requires_grad = bool(trainable)

    @torch.jit.unused
    def get_input_spec(self):
        return self.input_spec


class AbstractHead(AbstractDecoder):
    def __init__(self, input_spec: FeatureMapsSpecification):
        super().__init__(input_spec)

    @abstractmethod
    def forward(
        self, feature_maps: List[Tensor], output_size: Union[Tuple[int, int], torch.Size, None] = None
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Tensor], Mapping[str, Tensor]]:
        ...

    @torch.jit.unused
    def apply_to_final_layer(self, func: Callable[[nn.Module], None]):
        """
        Apply function to the final layer of the head.

        Typically, you can use this method to apply custom initialization
        to the final layer of the head.

        :param func: Function to apply to the final prediction layer. If head contains
        supervision layers, function can be applied to them as well.

        """
        raise NotImplementedError("This method is not implemented in class " + self.__class__.__name__)
