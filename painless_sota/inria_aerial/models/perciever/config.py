from dataclasses import asdict, dataclass, fields
from typing import Generic, Optional, TypeVar
from typing import Tuple

from painless_sota.inria_aerial.data.functional import as_tuple_of_two


def _base_kwargs(config, base_class, exclude):
    base_field_names = [field.name for field in fields(base_class) if field.name not in exclude]
    return {k: v for k, v in asdict(config).items() if k in base_field_names}


@dataclass
class EncoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    num_cross_attention_layers: int = 1
    first_cross_attention_layer_shared: bool = False
    cross_attention_widening_factor: int = 1
    num_self_attention_heads: int = 8
    num_self_attention_qk_channels: Optional[int] = None
    num_self_attention_v_channels: Optional[int] = None
    num_self_attention_layers_per_block: int = 8
    num_self_attention_blocks: int = 1
    first_self_attention_block_shared: bool = True
    self_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02
    freeze: bool = False
    attention_residual: bool = True

    def base_kwargs(self, exclude=("freeze",)):
        return _base_kwargs(self, EncoderConfig, exclude)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_size: Tuple[int, int] = (224, 224)
    input_channels: int = 3
    num_frequency_bands: int = 64
    include_positions: bool = False
    image_channels_before_concat: Optional[int] = None
    num_output_channels: Optional[int] = None

    type: str = "learnable"

    @property
    def image_shape(self):
        image_size = as_tuple_of_two(self.image_size)
        return tuple([*image_size, self.input_channels])


@dataclass
class DecoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02
    freeze: bool = False
    attention_residual: bool = False

    def base_kwargs(self, exclude=("freeze",)):
        return _base_kwargs(self, DecoderConfig, exclude)


@dataclass
class SegmentationDecoderConfig(DecoderConfig):
    num_classes: int = 10
    use_supervision: bool = True

    type: str = "same_input"


E = TypeVar("E", bound=EncoderConfig)
D = TypeVar("D", bound=DecoderConfig)


@dataclass
class PerceiverConfig(Generic[E, D]):
    encoder: E
    decoder: D
    num_latents: int
    num_latent_channels: int
    activation_checkpointing: bool = False
    activation_offloading: bool = False
    output_name: Optional[str] = None
