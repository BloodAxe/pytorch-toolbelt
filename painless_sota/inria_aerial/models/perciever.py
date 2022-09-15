import abc
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_toolbelt.utils import describe_outputs, count_parameters, to_numpy
from torch import Tensor, nn
from transformers import PerceiverConfig, apply_chunking_to_forward
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions
from transformers.models.perceiver.modeling_perceiver import (
    build_position_encoding,
    PerceiverMLP,
    PostprocessorType,
    PerceiverModelOutput,
    PerceiverTrainablePositionEncoding,
    PreprocessorType,
)
from transformers.utils import ModelOutput


class AbstractPreprocessor(nn.Module):
    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
        raise NotImplementedError()


@dataclass
class PerceiverDecoderOutput(ModelOutput):
    """
    Base class for Perceiver decoder outputs, with potential cross-attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Output of the basic decoder.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    logits: torch.FloatTensor = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class PerceiverAbstractDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Perceiver abstract decoder."""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    @torch.jit.ignore
    def num_query_channels(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, query, z, query_mask=None):
        raise NotImplementedError


class PerceiverEmbeddings(nn.Module):
    """Construct the latent embeddings."""

    def __init__(self, config):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.d_latents))

    def forward(self, batch_size: int):
        return self.latents.expand(batch_size, -1, -1)  # Thanks, Phil Wang


class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PerceiverSelfOutput(nn.Module):
    def __init__(self, config, input_channels, output_channels):
        super().__init__()
        self.dense = nn.Linear(input_channels, output_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states


class PerceiverAttention(nn.Module):
    """Attention module, including a dense block."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        use_query_residual=True,
    ):
        super().__init__()
        # MultiHead attention
        if is_cross_attention and qk_channels is None:
            if config.cross_attention_shape_for_attention == "q":
                qk_channels = q_dim
            elif config.cross_attention_shape_for_attention == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {config.cross_attention_shape_for_attention} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels
        self.self = PerceiverSelfAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
        )
        # dense block
        output_channels = None
        if is_cross_attention:
            output_channels = q_dim
        else:
            if output_channels is None:
                output_channels = v_channels
        self.output = PerceiverSelfOutput(config, input_channels=self.self.v_channels, output_channels=output_channels)
        self.use_query_residual = use_query_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )

        # Output projection
        attention_output = self.output(self_outputs[0])

        # Optionally include a residual to the original queries.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output


class PerceiverImagePreprocessor(nn.Module):
    """
    Image preprocessing for Perceiver Encoder.

    Note: the *out_channels* argument refers to the output channels of a convolutional layer, if *prep_type* is set to
    "conv1x1" or "conv". If one adds absolute position embeddings, one must make sure the *num_channels* of the
    position encoding kwargs are set equal to the *out_channels*.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        prep_type (`str`, *optional*, defaults to `"conv"`):
            Preprocessing type. Can be "conv1x1", "conv", "patches", "pixels".
        spatial_downsample (`int`, *optional*, defaults to 4):
            Spatial downsampling factor.
        temporal_downsample (`int`, *optional*, defaults to 1):
            Temporal downsampling factor (only relevant in case a time dimension is present).
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Position encoding type. Can be "fourier" or "trainable".
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input.
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        conv_after_patching (`bool`, *optional*, defaults to `False`):
            Whether to apply a convolutional layer after patching.
        conv_after_patching_in_channels (`int`, *optional*, defaults to 54):
            Number of channels in the input of the convolutional layer after patching.
        conv2d_use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batch normalization in the convolutional layer.
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(
        self,
        config,
        prep_type="conv1x1",
        in_channels: int = 3,
        out_channels: int = 64,
        conv_after_patching: bool = False,
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        project_pos_dim: int = -1,
        fourier_position_encoding_kwargs={},
    ):
        super().__init__()
        self.config = config

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channels = in_channels
        self.prep_type = prep_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        if self.prep_type == "conv1x1":
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
            )
        elif self.prep_type == "pixels":
            pass

        # Position embeddings
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type="fourier",
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs,
        )

    @property
    def num_channels(self) -> int:
        # Let's assume that the number of resolutions (in the context of image preprocessing)
        # of the input data is 2 or 3 depending on whether we are processing image or video respectively.
        # In this case, for convenience, we will declare is_temporal variable,
        # which will show whether the data has a temporal dimension or not.
        is_temporal = self.position_embeddings.num_dimensions > 2

        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim

        # inputs
        if self.conv_after_patching or self.prep_type in ("conv1x1", "conv"):
            inp_dim = self.out_channels
        elif self.prep_type == "pixels":
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == "patches":
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample**2
                if is_temporal:
                    inp_dim *= self.temporal_downsample

        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs: torch.Tensor, pos: torch.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.

        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc
        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        if self.prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)

        # move channels to last dimension, as the _build_network_inputs method below expects this
        inputs = torch.moveaxis(inputs, 1, -1)

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)

        return inputs, inputs_without_pos


class FourierPixelImagePreprocessor(nn.Module):
    def __init__(
        self,
        config,
        in_channels: int = 3,
        num_channels: int = -1,
        project_pos_dim: int = -1,
        fourier_position_encoding_kwargs={},
    ):
        super().__init__()
        self.config = config

        self.in_channels = in_channels
        self.project_input_pos_channels = num_channels

        # Position embeddings
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type="fourier",
            out_channels=None,
            project_pos_dim=project_pos_dim,
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs,
        )
        self.project_input_pos = (
            nn.Linear(self.in_channels + self.pos_emb_num_channels, self.project_input_pos_channels)
            if self.project_input_pos_channels > 0
            else nn.Identity()
        )

    @property
    @torch.jit.ignore
    def pos_emb_num_channels(self) -> int:
        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        return pos_dim

    @property
    @torch.jit.ignore
    def num_channels(self) -> int:
        # position embedding
        if self.project_input_pos_channels > 0:
            return self.project_input_pos_channels

        return self.in_channels + self.pos_emb_num_channels

    def _build_network_inputs(self, inputs: torch.Tensor, pos: torch.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.

        This method expects the inputs to always have channels as last dimension.

        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])

        inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        inputs_with_pos = self.project_input_pos(inputs_with_pos)
        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        # move channels to last dimension, as the _build_network_inputs method below expects this
        inputs = torch.moveaxis(inputs, 1, -1)
        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        return inputs, inputs_without_pos


class PerceiverBasicDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a
    cross-attention operation, in which the latents produce keys and values.

    The shape of the output of this class depends on how one defines the output queries (also called decoder queries).

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_num_channels (`int`, *optional*):
            The number of channels in the output. Will only be used in case *final_project* is set to `True`.
        position_encoding_type (`str`, *optional*, defaults to "trainable"):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
        output_index_dims (`int`, *optional*):
            The number of dimensions of the output queries. Ignored if 'position_encoding_type' == 'none'.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels of the decoder queries. Ignored if 'position_encoding_type' == 'none'.
        qk_channels (`int`, *optional*):
            The number of channels of the queries and keys in the cross-attention layer.
        v_channels (`int`, *optional*):
            The number of channels of the values in the cross-attention layer.
        num_heads (`int`, *optional*, defaults to 1):
            The number of attention heads in the cross-attention layer.
        widening_factor (`int`, *optional*, defaults to 1):
            The widening factor of the cross-attention layer.
        use_query_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a residual connection between the query and the output of the cross-attention layer.
        concat_preprocessed_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the preprocessed input to the query.
        final_project (`bool`, *optional*, defaults to `True`):
            Whether to project the output of the cross-attention layer to a target dimension.
        position_encoding_only (`bool`, *optional*, defaults to `False`):
            Whether to only use this class to define output queries.
    """

    def __init__(
        self,
        config: PerceiverConfig,
        output_num_channels: int,
        position_encoding_type: Optional[str] = "trainable",
        # The following 2 arguments are ignored if position_encoding_type == 'none':
        output_index_dims: Optional[int] = None,
        num_channels: Optional[int] = 128,
        subsampled_index_dims: Optional[int] = None,
        qk_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        num_heads: Optional[int] = 1,
        widening_factor: Optional[int] = 1,
        use_query_residual: Optional[bool] = False,
        concat_preprocessed_input: Optional[bool] = False,
        final_project: Optional[bool] = True,
        position_encoding_only: Optional[bool] = False,
        **position_encoding_kwargs,
    ) -> None:
        super().__init__()

        self.output_num_channels = output_num_channels
        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_position_encodings = None
        self.position_encoding_type = position_encoding_type
        self.position_encoding_kwargs = position_encoding_kwargs
        if position_encoding_type != "none":
            self.output_position_encodings, self.positions_projection = build_position_encoding(
                position_encoding_type=position_encoding_type, **position_encoding_kwargs
            )

        self.output_index_dims = output_index_dims
        self.num_channels = num_channels
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self.subsampled_index_dims = subsampled_index_dims
        self.concat_preprocessed_input = concat_preprocessed_input
        self.final_project = final_project
        self.position_encoding_only = position_encoding_only

        # for multimodal autoencoding, we don't need the decoder cross-attention and final layer
        # so then we will set position_encoding_only to True
        if not self.position_encoding_only:
            self.decoding_cross_attention = PerceiverLayer(
                config,
                is_cross_attention=True,
                qk_channels=qk_channels,
                v_channels=v_channels,
                num_heads=num_heads,
                q_dim=num_channels,
                kv_dim=config.d_latents,
                widening_factor=widening_factor,
                use_query_residual=use_query_residual,
            )
            self.final_layer = nn.Linear(num_channels, output_num_channels) if final_project else nn.Identity()

    @torch.jit.ignore
    def num_query_channels(self) -> int:
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError(
                "You cannot calculate number of decoder query channels when position_encoding_type is set to none"
            )
        if self.position_encoding_only:
            if "project_pos_dim" in self.position_encoding_kwargs:
                return self.position_encoding_kwargs["project_pos_dim"]
            return self.output_position_encodings.output_size()
        if self.final_project:
            return self.output_num_channels
        return self.num_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if self.position_encoding_type == "none":  # Queries come from elsewhere
            raise ValueError("You cannot construct decoder queries when position_encoding_type is set to none")
        if subsampled_points is not None:
            # subsampled_points are the indices if the inputs would be flattened
            # however, the inputs aren't flattened, that's why we use unravel_index
            # to get the indices for the unflattened array
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            indices = list(
                torch.from_numpy(x) for x in np.unravel_index(subsampled_points.cpu(), self.output_index_dims)
            )
            pos = torch.stack(indices, dim=1)
            batch_size = inputs.shape[0]
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / torch.tensor(self.output_index_dims)[None, :]
            pos = torch.broadcast_to(pos[None], [batch_size, pos.shape[0], pos.shape[1]])
            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(
                    self.output_index_dims, batch_size=batch_size, device=inputs.device, pos=pos
                )

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)
            pos_emb = torch.reshape(pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            batch_size = inputs.shape[0]
            index_dims = inputs.shape[2:]

            # Construct the position encoding.
            if self.position_encoding_type == "trainable":
                pos_emb = self.output_position_encodings(batch_size)
            elif self.position_encoding_type == "fourier":
                pos_emb = self.output_position_encodings(index_dims, batch_size, device=inputs.device)

            # Optionally project them to a target dimension.
            pos_emb = self.positions_projection(pos_emb)

        if self.concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError("Value is required for inputs_without_pos if concat_preprocessed_input is True")
            pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)

        return pos_emb

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> PerceiverDecoderOutput:
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        cross_attentions = () if output_attentions else None

        layer_outputs = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
            output_attentions=output_attentions,
        )
        output = layer_outputs[0]

        # print(f"decoding_cross_attention")
        # print("mean", hidden_states.mean(dim=(0, 1)))
        # print("std ", output.std(dim=(0, 1)))

        if output_attentions:
            cross_attentions = cross_attentions + (layer_outputs[1],)

        logits = self.final_layer(output)

        return PerceiverDecoderOutput(logits=logits, cross_attentions=cross_attentions)


class PerceiverSegmentationDecoder(PerceiverAbstractDecoder):
    """Cross-attention based optical flow decoder."""

    def __init__(self, config, output_image_shape, output_num_channels=2, **decoder_kwargs):
        super().__init__()

        self.output_image_shape = output_image_shape
        self.output_num_channels = output_num_channels

        self.output_position_encodings, self.positions_projection = build_position_encoding(
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs=decoder_kwargs["fourier_position_encoding_kwargs"],
        )

        self.decoder = PerceiverBasicDecoder(
            config,
            position_encoding_type="none",
            concat_preprocessed_input=False,
            num_channels=258,
            output_num_channels=output_num_channels,
        )

    @property
    @torch.jit.ignore
    def num_query_channels(self) -> int:
        return 258

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        pos_emb = self.output_position_encodings(self.output_image_shape, inputs.shape[0], device=inputs.device)
        pos_emb = self.positions_projection(pos_emb)
        return pos_emb

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> PerceiverDecoderOutput:
        decoder_outputs = self.decoder(query=query, z=z, output_attentions=output_attentions)
        preds = decoder_outputs.logits
        # Output flow and rescale.
        preds = preds.reshape([preds.shape[0]] + list(self.output_image_shape) + [preds.shape[-1]])
        return PerceiverDecoderOutput(logits=preds, cross_attentions=decoder_outputs.cross_attentions)


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(self, config, kv_dim=None):
        super().__init__()
        self.config = config

        # Check that we can use multihead-attention with these shapes.
        if config.d_latents % config.num_self_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_self_attend_heads ({config.num_self_attention_heads})."
            )
        if config.d_latents % config.num_cross_attention_heads != 0:
            raise ValueError(
                f"num_z_channels ({config.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({config.num_cross_attention_heads})."
            )

        # Construct the cross attention layer.
        self.cross_attention = PerceiverLayer(
            config,
            is_cross_attention=True,
            qk_channels=config.qk_channels,
            v_channels=config.v_channels,
            num_heads=config.num_cross_attention_heads,
            q_dim=config.d_latents,
            kv_dim=kv_dim,
            widening_factor=config.cross_attention_widening_factor,
            use_query_residual=config.use_query_residual,
        )

        # Construct a single block of self-attention layers.
        # We get deeper architectures by applying this block more than once.
        self_attention_layers = []
        for _ in range(config.num_self_attends_per_block):
            layer = PerceiverLayer(
                config,
                is_cross_attention=False,
                qk_channels=config.qk_channels,
                v_channels=config.v_channels,
                num_heads=config.num_self_attention_heads,
                q_dim=config.d_latents,
                kv_dim=config.d_latents,
                widening_factor=config.self_attention_widening_factor,
            )
            self_attention_layers.append(layer)

        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        # Apply the cross-attention between the latents (hidden_states) and inputs:
        layer_outputs = self.cross_attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            inputs=inputs,
            inputs_mask=inputs_mask,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0]
        # print("hidden_states after cross attention")
        # print("mean", hidden_states.mean(dim=(0, 1)))
        # print("std ", hidden_states.std(dim=(0, 1)))

        if output_attentions:
            all_cross_attentions = all_cross_attentions + (layer_outputs[1],)

        # Apply the block of self-attention layers more than once:
        for j in range(self.config.num_blocks):
            for i, layer_module in enumerate(self.self_attends):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                # print(f"hidden_states after {j}/{i} layer")
                # print("mean", hidden_states.mean(dim=(0, 1)))
                # print("std ", hidden_states.std(dim=(0, 1)))

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PerceiverModel(nn.Module):
    def __init__(
        self,
        config: PerceiverConfig,
        decoder: PerceiverAbstractDecoder,
        input_preprocessor: PreprocessorType,
        output_postprocessor: PostprocessorType = None,
    ):
        super().__init__()
        self.config = config

        self.input_preprocessor = input_preprocessor
        self.output_postprocessor = output_postprocessor
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(
            config, kv_dim=input_preprocessor.num_channels if input_preprocessor is not None else config.d_model
        )
        self.decoder = decoder

        # Initialize weights and apply final processing
        # self.post_init()
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if hasattr(module, "latents"):
                torch.nn.init.trunc_normal_(module.latents, mean=0, std=self.config.initializer_range)
                # torch.nn.init.orthogonal_(module.latents, gain=1)
                # torch.nn.init.orthogonal_(module.latents, gain=1)
                # torch.nn.init.kaiming_normal_(module.latents, a=0.01, mode="fan_in")
                # module.latents.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                torch.nn.init.trunc_normal_(module.weight, mean=0, std=self.config.initializer_range)
                # torch.nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
                # torch.nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_in")
                # if module.bias is not None:
                #    torch.nn.init.zeros_(module.bias)
                pass
            elif hasattr(module, "position_embeddings") and isinstance(module, PerceiverTrainablePositionEncoding):
                module.position_embeddings.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.ParameterDict):
                for modality in module.keys():
                    module[modality].data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        inputs: torch.FloatTensor,
    ) -> PerceiverModelOutput:

        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states

        inputs, inputs_without_pos = self.input_preprocessor(inputs)

        # a = to_numpy(inputs[0, 0])
        # b = to_numpy(inputs[0, -1])
        # c = to_numpy(inputs[0, 1000])
        # d = to_numpy(inputs[0, -1000])
        # s = np.stack([a, b, c, d])

        batch_size, seq_length, _ = inputs.size()
        embedding_output = self.embeddings(batch_size=batch_size)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=None,
            head_mask=None,
            inputs=inputs,
            inputs_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]

        logits = None
        if self.decoder:
            decoder_query = self.decoder.decoder_query(
                inputs, modality_sizes=None, inputs_without_pos=inputs_without_pos
            )
            decoder_outputs = self.decoder(
                decoder_query,
                z=sequence_output,
                query_mask=None,
                output_attentions=output_attentions,
            )
            logits = decoder_outputs.logits

            if self.output_postprocessor:
                logits = self.output_postprocessor(logits)

        return PerceiverModelOutput(
            logits=logits,
        )


class PerceiverForSegmentation(nn.Module):
    def __init__(self, config, num_classes, output_name):
        super().__init__()

        fourier_position_encoding_kwargs_preprocessor = dict(
            num_bands=64,
            max_resolution=(config.train_size[0] // 2, config.train_size[1] // 2),
            sine_only=False,
            concat_pos=True,
        )
        fourier_position_encoding_kwargs_decoder = dict(
            concat_pos=True,
            max_resolution=(config.train_size[0] // 2, config.train_size[1] // 2),
            num_bands=64,
            sine_only=False,
        )

        image_preprocessor = FourierPixelImagePreprocessor(
            config,
            in_channels=27,
            num_channels=-1,
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_preprocessor,
        )

        mask_decoder = PerceiverSegmentationDecoder(
            config,
            num_channels=image_preprocessor.num_channels,
            output_image_shape=(config.train_size[0] // 2, config.train_size[1] // 2),
            # decoder kwargs
            use_query_residual=False,
            output_num_channels=num_classes,
            # We query the decoder using the first frame features
            # rather than a standard decoder position encoding.
            position_encoding_type="fourier",
            fourier_position_encoding_kwargs=fourier_position_encoding_kwargs_decoder,
        )

        self.perceiver = PerceiverModel(config, input_preprocessor=image_preprocessor, decoder=mask_decoder)
        self.output_name = output_name
        
    @classmethod
    def extract_image_patches(cls, x, kernel, stride=1, dilation=1):

        # Do TF 'SAME' Padding
        b, c, h, w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

        return patches.view(b, -1, patches.shape[-2], patches.shape[-1])

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tensor]:
        patches = self.extract_image_patches(image, kernel=3, stride=2)

        outputs = self.perceiver(
            inputs=patches,
        )

        mask = F.interpolate(
            torch.moveaxis(outputs.logits, -1, 1), scale_factor=2, mode="bilinear", align_corners=False
        )
        if self.output_name is not None:
            return {self.output_name: mask}
        else:
            return mask

    @classmethod
    def from_config(cls, config):
        return PerceiverForSegmentation(
            config=PerceiverConfig(
                train_size=list(config.train_size),
                d_model=322,
                d_latents=config.d_latents,
                num_latents=config.num_latents,
                num_self_attention_heads=config.num_self_attention_heads,
                num_self_attends_per_block=config.num_self_attends_per_block,
                num_cross_attention_heads=config.num_cross_attention_heads,
                use_query_residual=False,
                initializer_range=0.15,
            ),
            output_name=config.output_name,
            num_classes=config.num_classes,
        )


if __name__ == "__main__":
    config = PerceiverConfig(
        train_size=[256, 256],
        d_model=384,
        d_latents=512,
        num_latents=2048,
        num_cross_attention_heads=1,
        num_self_attention_heads=8,
        num_self_attends_per_block=12,
        use_query_residual=False,
        initializer_range=0.15,
    )
    model = PerceiverForSegmentation(config, num_classes=10,output_name=None).cuda()
    input = torch.randn((1, 3, 256, 256)).cuda()

    input[:, 0, :128, :128] = 1
    input[:, 0, 128:, :128] = 0
    input[:, 0, :128, 128:] = 0
    input[:, 0, 128:, 128:] = -1

    input[:, 1, :128, :128] = 0
    input[:, 1, 128:, :128] = 1
    input[:, 1, :128, 128:] = -1
    input[:, 1, 128:, 128:] = 0

    input[:, 2, :128, :128] = 0
    input[:, 2, 128:, :128] = -1
    input[:, 2, :128, 128:] = 1
    input[:, 2, 128:, 128:] = 1

    with torch.no_grad():
        with torch.cuda.amp.autocast(True):
            output = model(input)

    traced_model = torch.jit.trace(
        model.eval().cuda(),
        example_inputs=input,
        strict=False,
    )

    classmap = to_numpy(output.argmax(dim=1))[0]

    plt.figure()
    plt.imshow(classmap)
    plt.tight_layout()
    plt.axis("off")
    plt.show()

    print(count_parameters(model))
    print(describe_outputs(output))
    print(np.unique(classmap.flatten()), np.bincount(classmap.flatten()))

    probs = to_numpy(output.softmax(dim=1))
    print(probs[0, :, 10, 100])
    print(probs[0, :, 10, 10])
    print(probs[0, :, 100, 100])
    print(probs[0, :, 100, 10])
    print(probs[0, :, 200, 200])
