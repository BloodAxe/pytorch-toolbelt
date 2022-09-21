from abc import abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Optional
from typing import Tuple

from pytorch_toolbelt.modules import ACT_GELU

__all__ = [
    "PostprocessorConfig",
    "PerceiverConfig",
    "EncoderConfig",
    "PositionEncodingConfig",
    "LearnablePositionEncodingConfig",
    "DecoderConfig",
    "DecoderQueryConfig",
    "EncoderInputQueryConfig",
    "FourierPositionEncodingQueryConfig",
    "LearnableConvPreprocessorConfig",
    "Depth2SpacePostprocessorConfig",
    "Space2DepthPreprocessorConfig",
    "FourierPositionEncodingConfig",
    "PreprocessorConfig",
]


@dataclass
class PreprocessorConfig:
    spatial_shape: Tuple[int, int]
    num_input_channels: int
    num_output_channels: Optional[int] = None

    @property
    @abstractmethod
    def output_spatial_shape(self) -> Tuple[int, int]:
        raise NotImplementedError()


@dataclass
class Space2DepthPreprocessorConfig(PreprocessorConfig):
    factor: int = 4
    with_bn: bool = True
    activation: Optional[str] = ACT_GELU
    kernel_size: int = 3


@dataclass
class LearnableConvPreprocessorConfig(PreprocessorConfig):
    activation: Optional[str] = ACT_GELU


@dataclass
class PositionEncodingConfig:
    pass


@dataclass
class FourierPositionEncodingConfig(PositionEncodingConfig):
    num_frequency_bands: int = 64
    include_positions: bool = True
    num_output_channels: Optional[int] = None


@dataclass
class LearnablePositionEncodingConfig(PositionEncodingConfig):
    num_output_channels: int
    init_scale: float = 0.02


@dataclass
class EncoderConfig:
    num_latents: int = 1024
    num_latent_channels: int = 512

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
    attention_residual: bool = True

    activation: str = ACT_GELU
    activation_checkpointing: bool = True
    activation_offloading: bool = False


@dataclass
class DecoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02
    attention_residual: bool = False

    activation: str = ACT_GELU
    activation_checkpointing: bool = True
    activation_offloading: bool = False


@dataclass
class DecoderQueryConfig:
    pass


@dataclass
class EncoderInputQueryConfig(DecoderQueryConfig):
    pass


@dataclass
class FourierPositionEncodingQueryConfig(DecoderQueryConfig):
    pass


@dataclass
class PostprocessorConfig:
    pass


@dataclass
class Depth2SpacePostprocessorConfig(PostprocessorConfig):
    num_output_channels: int
    factor: int = 4
    output_name: Optional[str] = None
    activation: str = ACT_GELU


@dataclass
class PerceiverConfig:
    preprocessor: PreprocessorConfig
    position_encoding: PositionEncodingConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    output_query: DecoderQueryConfig
    postprocessor: PostprocessorConfig
