import math
from dataclasses import asdict, dataclass, fields
from typing import Generic, Optional, TypeVar
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from fairscale.nn import checkpoint_wrapper

from painless_sota.inria_aerial.data.functional import as_tuple_of_two
from pytorch_toolbelt.utils import count_parameters, describe_outputs
from torch import Tensor


__all__ = ["PercieverIOForSegmentation"]


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

    def base_kwargs(self, exclude=("freeze",)):
        return _base_kwargs(self, EncoderConfig, exclude)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 64


@dataclass
class DecoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02
    freeze: bool = False

    def base_kwargs(self, exclude=("freeze",)):
        return _base_kwargs(self, DecoderConfig, exclude)


@dataclass
class SegmentationDecoderConfig(DecoderConfig):
    num_classes: int = 10
    num_frequency_bands: int = 64


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
    output_name: str = None


@dataclass
class ClassificationDecoderConfig(DecoderConfig):
    num_output_queries: int = 1
    num_output_query_channels: int = 256
    num_classes: int = 100


def _base_kwargs(config, base_class, exclude):
    base_field_names = [field.name for field in fields(base_class) if field.name not in exclude]
    return {k: v for k, v in asdict(config).items() if k in base_field_names}


class Sequential(nn.Sequential):
    def forward(self, *x):
        for module in self:
            if type(x) == tuple:
                x = module(*x)
            else:
                x = module(x)
        return x


def freeze(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError(f"num_qk_channels {num_qk_channels} must be divisible by num_heads {num_heads}")

        if num_v_channels % num_heads != 0:
            raise ValueError(f"num_v_channels {num_v_channels} must be divisible by num_heads {num_heads}")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head cross-attention (see `MultiHeadAttention` for details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """Multi-head attention of query input `x_q` to key/value input (`x_kv`) after (separately) applying layer
        normalization to these inputs."""
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head self-attention (see `MultiHeadAttention` and for details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        """Multi-head attention of input `x` to itself after applying layer normalization to the input."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class CrossAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        attention_residual: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        super().__init__(
            Residual(cross_attn, dropout) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor), dropout),
        )


class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=dropout,
        )
        super().__init__(
            Residual(self_attn, dropout),
            Residual(MLP(num_channels, widening_factor), dropout),
        )


class SelfAttentionBlock(Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                widening_factor=widening_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

        if activation_checkpointing:
            layers = [checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) for layer in layers]

        super().__init__(*layers)


class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels),
        )


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels: int):
        """Transforms and position-encodes task-specific input to generic encoder input.

        :param num_input_channels: Number of channels of the generic encoder input produced by this adapter.
        """
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, x):
        raise NotImplementedError()


class OutputAdapter(nn.Module):
    def __init__(self, output_query: Tensor, init_scale: float):
        """Transforms generic decoder cross-attention output to task-specific output.

        :param output_query: Output query prototype (does not include batch dimension) used as query input to
            generic decoder cross-attention.
        :param init_scale: Output query parameter initialization scale.
        """
        super().__init__()
        self._output_query = nn.Parameter(output_query)
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self._output_query.normal_(0.0, init_scale)

    @property
    def num_output_query_channels(self):
        return self._output_query.shape[-1]

    def output_query(self, x):
        return repeat(self._output_query, "... -> b ...", b=x.shape[0])

    def forward(self, x):
        raise NotImplementedError()


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_output_queries: int = 1,
        num_output_query_channels: Optional[int] = None,
        init_scale: float = 0.02,
    ):

        if num_output_query_channels is None:
            num_output_query_channels = num_classes

        super().__init__(
            output_query=torch.empty(num_output_queries, num_output_query_channels), init_scale=init_scale
        )
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class SegmentationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        num_classes: int,
        num_frequency_bands: int,
        image_shape,
        num_output_query_channels: Optional[int] = None,
        init_scale: float = 0.02,
    ):
        *spatial_shape, num_image_channels = image_shape

        # create encodings for single example
        pos = self._positions(spatial_shape)
        enc = self._position_encodings(pos, num_frequency_bands)
        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        super().__init__(output_query=enc, init_scale=init_scale)
        self.linear = nn.Linear(self.num_output_query_channels, num_classes)
        self.spatial_shape = spatial_shape

    def _init_parameters(self, init_scale):
        pass

    @classmethod
    def _positions(cls, spatial_shape, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].

        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        coords = [torch.linspace(v_min, v_max, steps=s) for s in spatial_shape]
        return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape))

    @classmethod
    def _position_encodings(
        cls,
        p: Tensor,
        num_frequency_bands,
        max_frequencies: Optional[Tuple[int, ...]] = None,
        include_positions: bool = True,
    ) -> Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.

        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            max_frequencies = p.shape[:-1]

        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, num_frequency_bands, device=p.device) for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

        if include_positions:
            encodings.append(p)

        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        return torch.cat(encodings, dim=-1)

    def output_query(self, x):
        b, *d = x.shape
        return repeat(self._output_query, "... -> b ...", b=b)

    def forward(self, x):
        y = self.linear(x)
        b, spatial_flatten, channels = y.shape

        return torch.moveaxis(y.view([b] + self.spatial_shape + [channels]), -1, 1)


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input
            of shape (B, M, C) where B is the batch size, M the input sequence length and C the number of
            key/value input channels. C is determined by the `num_input_channels` property of the
            `input_adapter`.
        :param num_latents: Number of latent variables (N).
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights
            with subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights
            with subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        self.input_adapter = input_adapter

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError("num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            return (
                checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) if activation_checkpointing else layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
            )

        self.cross_attn_n = cross_attn()
        self.self_attn_n = self_attn()

        if self.first_cross_attention_layer_shared or num_cross_attention_layers == 1:
            self.cross_attn_1 = self.cross_attn_n
        else:
            self.cross_attn_1 = cross_attn()

        if self.first_self_attention_block_shared or num_self_attention_blocks == 1:
            self.self_attn_1 = self.self_attn_n
        else:
            self.self_attn_1 = self_attn()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            self.latent.normal_(0.0, init_scale)
            _init_parameters(self, init_scale)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn_1(x_latent, x, pad_mask)
        x_latent = self.self_attn_1(x_latent)

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = self.cross_attn_n(x_latent, x, pad_mask)
            x_latent = self.self_attn_n(x_latent)

        return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        output_adapter: OutputAdapter,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        cross_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder cross-attention output of shape (B, O, F) to task-specific
            output. B is the batch size, O the output sequence length and F the number of cross-attention output
            channels. F is determined by the `num_output_query_channels` property of the `output_adapter`.
        :param num_latent_channels: Number of latent channels (C_latent) as produced by a Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layers and residuals.
        :param init_scale: Standard deviation for random normal initialization of parameters.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        :param activation_offloading: If True, offloads checkpointed activations to CPU.
        """
        super().__init__()

        cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_adapter.num_output_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout,
        )

        if activation_checkpointing:
            cross_attn = checkpoint_wrapper(cross_attn, offload_to_cpu=activation_offloading)

        self.cross_attn = cross_attn
        self.output_adapter = output_adapter
        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            _init_parameters(self, init_scale)

    def forward(self, x):
        output_query = self.output_adapter.output_query(x)
        output = self.cross_attn(output_query, x)
        return self.output_adapter(output)


class PerceiverIO(Sequential):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)

    @property
    def encoder(self):
        return self[0]

    @property
    def decoder(self):
        return self[1]


def _init_parameters(module, init_scale):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)


class ImageInputAdapter(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
        *self.spatial_shape, num_image_channels = image_shape
        self.image_shape = image_shape
        self.num_frequency_bands = num_frequency_bands

        super().__init__(num_input_channels=num_image_channels + self._num_position_encoding_channels())

        # create encodings for single example
        pos = self._positions()
        enc = self._position_encodings(pos)

        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _positions(self, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].

        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        coords = [torch.linspace(v_min, v_max, steps=s) for s in self.spatial_shape]
        return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(self.spatial_shape))

    def _position_encodings(
        self, p: Tensor, max_frequencies: Optional[Tuple[int, ...]] = None, include_positions: bool = True
    ) -> Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.

        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            max_frequencies = p.shape[:-1]

        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, self.num_frequency_bands, device=p.device)
            for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

        if include_positions:
            encodings.append(p)

        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        return torch.cat(encodings, dim=-1)

    def _num_position_encoding_channels(self, include_positions: bool = True) -> int:
        return len(self.spatial_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, x):
        x = torch.moveaxis(x, 1, -1)  # Move BCHW to BHWC
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        x_enc = repeat(self.position_encoding, "... -> b ...", b=b)
        x = rearrange(x, "b ... c -> b (...) c")
        return torch.cat([x, x_enc], dim=-1)


class ImageClassifier(PerceiverIO):
    def __init__(self, config: PerceiverConfig[ImageEncoderConfig, ClassificationDecoderConfig]):
        input_adapter = ImageInputAdapter(
            image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
        )

        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **encoder_kwargs,
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=config.decoder.num_classes,
            num_output_queries=config.decoder.num_output_queries,
            num_output_query_channels=config.decoder.num_output_query_channels,
            init_scale=config.decoder.init_scale,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)


class PercieverIOForSegmentation(PerceiverIO):
    @classmethod
    def from_config(self, config):
        train_size = as_tuple_of_two(config.train_size)

        image_shape: Tuple[int, int, int] = list(train_size) + [3]
        return PercieverIOForSegmentation(
            PerceiverConfig(
                activation_checkpointing=config.activation_checkpointing,
                activation_offloading=config.activation_offloading,
                encoder=ImageEncoderConfig(
                    image_shape=tuple(image_shape),
                    num_cross_attention_heads=1,
                    num_self_attention_heads=config.num_self_attention_heads,
                    num_self_attention_layers_per_block=config.num_self_attends_per_block,
                    dropout=0.1,
                ),
                decoder=SegmentationDecoderConfig(
                    num_classes=config.num_classes, num_cross_attention_heads=1, dropout=0.1
                ),
                num_latents=config.num_latents,
                num_latent_channels=config.d_latents,
                output_name=config.output_name,
            )
        )

    def __init__(self, config: PerceiverConfig[ImageEncoderConfig, SegmentationDecoderConfig]):
        input_adapter = ImageInputAdapter(
            image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
        )

        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **encoder_kwargs,
        )
        output_adapter = SegmentationOutputAdapter(
            num_classes=config.decoder.num_classes,
            num_frequency_bands=config.decoder.num_frequency_bands,
            init_scale=config.decoder.init_scale,
            image_shape=config.encoder.image_shape,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            activation_offloading=config.activation_offloading,
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)
        self.output_name = config.output_name

    def forward(self, x):
        output = super().forward(x)
        if self.output_name is not None:
            return {self.output_name: output}
        else:
            return output


if __name__ == "__main__":
    model = (
        PercieverIOForSegmentation(
            PerceiverConfig(
                encoder=ImageEncoderConfig(image_shape=(256, 256, 3), num_cross_attention_heads=1, dropout=0.1),
                decoder=SegmentationDecoderConfig(num_classes=7, num_cross_attention_heads=1, dropout=0.1),
                num_latents=512,
                num_latent_channels=512,
                activation_checkpointing=False,
                activation_offloading=False,
            )
        )
        .eval()
        .cuda()
    )

    input = torch.randn((2, 3, 256, 256)).cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            output = model(input)
    print(count_parameters(model))
    print(describe_outputs(output))

    traced_model = torch.jit.trace(
        model.eval().cuda(),
        example_inputs=input,
        strict=False,
    )
