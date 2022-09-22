from dataclasses import asdict
from pprint import pprint
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat
from fairscale.nn import checkpoint_wrapper
from painless_sota.inria_aerial.models.perciever.attention import (
    CrossAttentionLayer,
    SelfAttentionBlock,
    _init_parameters,
)
from painless_sota.inria_aerial.models.perciever.config import (
    PerceiverConfig,
    Space2DepthPreprocessorConfig,
    LearnableConvPreprocessorConfig,
    FourierPositionEncodingConfig,
    Depth2SpacePostprocessorConfig,
    DecoderConfig,
    EncoderConfig,
    FourierPositionEncodingQueryConfig,
    EncoderInputQueryConfig,
)
from painless_sota.inria_aerial.models.perciever.decoder_query import (
    DecoderQuery,
    FourierPositionEncodingQuery,
    EncoderInputQuery,
)
from painless_sota.inria_aerial.models.perciever.position_encoding import (
    FourierPositionEncoding,
    PositionEncoding,
    PositionEncodingOutput,
)
from painless_sota.inria_aerial.models.perciever.postprocessor import Depth2SpacePostprocessor
from painless_sota.inria_aerial.models.perciever.preprocessors import (
    LearnableConvPreprocessor,
    Space2DepthPreprocessor,
    ImagePreprocessor,
)
from pytorch_toolbelt.modules import ACT_GELU
from pytorch_toolbelt.utils import count_parameters, describe_outputs, master_print
from torch import Tensor

__all__ = ["PercieverIOForSegmentation"]


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
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
        activation: str = ACT_GELU,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        attention_residual=True,
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
                num_kv_input_channels=num_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
                attention_residual=attention_residual,
                activation=activation,
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
                activation=activation,
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
        torch.nn.init.trunc_normal_(self.latent, 0.0, init_scale)
        _init_parameters(self, init_scale)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

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
        num_output_query_channels: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        cross_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation: str = ACT_GELU,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        attention_residual: bool = True,
    ):
        """Generic Perceiver IO decoder.

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
            num_q_input_channels=num_output_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            dropout=dropout,
            attention_residual=attention_residual,
            activation=activation,
        )

        if activation_checkpointing:
            cross_attn = checkpoint_wrapper(cross_attn, offload_to_cpu=activation_offloading)

        self.cross_attn = cross_attn
        self._init_parameters(init_scale)
        self._num_output_query_channels = num_output_query_channels

    @property
    def num_output_query_channels(self) -> int:
        return self._num_output_query_channels

    def _init_parameters(self, init_scale: float):
        _init_parameters(self, init_scale)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        output = self.cross_attn(q, kv)
        return output


class PercieverIOForSegmentation(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()

        if isinstance(config.preprocessor, Space2DepthPreprocessorConfig):
            preprocessor = Space2DepthPreprocessor(**asdict(config.preprocessor))
        elif isinstance(config.preprocessor, LearnableConvPreprocessorConfig):
            preprocessor = LearnableConvPreprocessor(**asdict(config.preprocessor))
        else:
            raise RuntimeError("Unsupported preprocessor type")

        if isinstance(config.position_encoding, FourierPositionEncodingConfig):
            position_encoding = FourierPositionEncoding(
                spatial_shape=preprocessor.output_spatial_shape,
                num_input_channels=preprocessor.num_output_channels,
                **asdict(config.position_encoding),
            )
        else:
            raise RuntimeError("Unsupported position encoding type")

        # Infer number of query channels
        # if config.encoder.num_cross_attention_qk_channels is None:
        #     config.encoder.num_cross_attention_qk_channels = position_encoding.num_output_channels
        #     master_print(
        #         f"Using value of position_encoding.num_output_channels ({position_encoding.num_output_channels}) "
        #         "to set config.encoder.num_cross_attention_qk_channels"
        #     )

        encoder = PerceiverEncoder(num_input_channels=position_encoding.num_output_channels, **asdict(config.encoder))

        if isinstance(config.output_query, FourierPositionEncodingQueryConfig):
            output_query = FourierPositionEncodingQuery(
                num_position_encoding_channels=position_encoding.num_position_encoding_channels,
                **asdict(config.output_query),
            )
        elif isinstance(config.output_query, EncoderInputQueryConfig):
            output_query = EncoderInputQuery(
                num_input_channels=position_encoding.num_output_channels, **asdict(config.output_query)
            )
        else:
            raise RuntimeError("Unsupported postprocessor config")

        decoder = PerceiverDecoder(
            num_latent_channels=config.encoder.num_latent_channels,
            num_output_query_channels=output_query.num_output_channels,
            **asdict(config.decoder),
        )

        if isinstance(config.postprocessor, Depth2SpacePostprocessorConfig):
            postprocessor = Depth2SpacePostprocessor(
                num_input_channels=output_query.num_output_channels,
                spatial_shape=preprocessor.output_spatial_shape,
                **asdict(config.postprocessor),
            )
        else:
            raise RuntimeError("Unsupported postprocessor config")

        self.preprocessor: ImagePreprocessor = preprocessor
        self.position_encoding: PositionEncoding = position_encoding
        self.encoder = encoder
        self.decoder = decoder
        self.output_query: DecoderQuery = output_query
        self.postprocessor = postprocessor
        pprint(config, indent=2)

    def forward(self, x: Tensor):
        x_pre = self.preprocessor(x)
        x: PositionEncodingOutput = self.position_encoding(x_pre)

        z = self.encoder(x.encoded_input)
        q = self.output_query(x=x, z=z)
        z = self.decoder(q, z)
        output = self.postprocessor(z)
        return output


if __name__ == "__main__":
    config = PerceiverConfig(
        preprocessor=Space2DepthPreprocessorConfig(
            spatial_shape=(512, 384), num_input_channels=3, factor=4, num_output_channels=64
        ),
        position_encoding=FourierPositionEncodingConfig(
            num_output_channels=None,
        ),
        encoder=EncoderConfig(
            num_latents=1024,
            num_latent_channels=512,
            num_cross_attention_heads=1,
            num_self_attention_heads=16,
            num_self_attention_layers_per_block=24,
            dropout=0.1,
            init_scale=0.05,
        ),
        decoder=DecoderConfig(
            num_cross_attention_heads=1,
            init_scale=0.05,
            dropout=0.1,
        ),
        output_query=EncoderInputQueryConfig(),
        postprocessor=Depth2SpacePostprocessorConfig(
            num_output_channels=1,
            factor=4,
        ),
    )
    model = PercieverIOForSegmentation(config).eval().cuda()

    input = torch.randn((2, 3, 512, 384)).cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            output = model(input)

    print(model)
    print(
        count_parameters(
            model,
            human_friendly=True,
            keys=["encoder", "decoder", "preprocessor", "postprocessor", "position_encoding", "output_query"],
        )
    )
    print(describe_outputs(output))
