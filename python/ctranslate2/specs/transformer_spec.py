"""Declares specification of the Transformer model."""

from typing import Optional, Tuple, Union

import numpy as np

from ctranslate2.specs import attention_spec, common_spec, model_spec


class TransformerEncoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        no_final_norm: bool = False,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        multi_query_attention: bool = False,
        # RoPE arguments
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1.0,
        rotary_base: float = 10000.0,
        original_max_position_embeddings: int = 0,
        max_position_embeddings: int = 0,
    ):
        self.multi_query_attention = multi_query_attention # Retained for potential internal logic
        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.embeddings_merge = np.dtype("int8").type(embeddings_merge)
        self.embeddings = [
            common_spec.EmbeddingsSpec() for _ in range(num_source_embeddings)
        ]
        self.scale_embeddings = True

        if not relative_position and not relative_attention_bias and rotary_dim is None:
            self.position_encodings = PositionEncoderSpec()
        
        if pre_norm and not no_final_norm:
            self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        if layernorm_embedding:
            self.layernorm_embedding = common_spec.LayerNormSpec(rms_norm=rms_norm)
        
        actual_num_heads_kv_encoder = None
        if multi_query_attention: # If MQA is true for the whole encoder stack
            actual_num_heads_kv_encoder = 1
        # elif num_heads_kv is not None: # If a specific num_heads_kv is passed for encoder
        #     actual_num_heads_kv_encoder = num_heads_kv

        self.layer = [
            TransformerEncoderLayerSpec(
                relative_position=relative_position,
                relative_attention_bias=relative_attention_bias,
                ffn_glu=ffn_glu,
                rms_norm=rms_norm,
                num_heads_kv=actual_num_heads_kv_encoder, # Pass effective num_heads_kv
                rotary_dim=rotary_dim,
                rotary_interleave=rotary_interleave,
                rotary_scaling_type=rotary_scaling_type,
                rotary_scaling_factor=rotary_scaling_factor,
                rotary_base=rotary_base,
                original_max_position_embeddings=original_max_position_embeddings,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ]

class TransformerDecoderSpec(model_spec.LayerSpec):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        with_encoder_attention: bool = True,
        no_final_norm: bool = False,
        project_in_out: bool = False, # Keep existing
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        alibi: bool = False,
        alibi_use_positive_positions: bool = False,
        scale_alibi: bool = False, 
        # RoPE arguments
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1.0,
        rotary_base: float = 10000.0,
        original_max_position_embeddings: int = 0,
        max_position_embeddings: int = 0,
        # Other structural arguments
        parallel_residual: bool = False, # Keep existing
        shared_layer_norm: bool = False, # Keep existing
        pre_post_layer_norm: bool = False, # Keep existing
        multi_query_attention: bool = False, 
        num_heads_kv: Optional[int] = None, 
        head_dim: Optional[int] = None, # Explicit head_dim for decoder
        sliding_window: Optional[int] = None, # Keep existing
        quant_type: Optional[common_spec.Quantization] = None, # Keep existing
        quant_group_size: Optional[int] = None, # Keep existing
        quant_bits: Optional[int] = None, # Keep existing
    ):
        self._config = {} # Initialize config dict
        if parallel_residual:
            if not pre_norm:
                raise ValueError("The GPT-J block expects a pre-norm architecture")
            if with_encoder_attention: # Assuming GPT-J like blocks don't have cross-attn
                raise ValueError("The GPT-J block does not have cross attention")

        # Determine effective num_heads_kv for decoder's attention layers
        actual_num_heads_kv_decoder = num_heads_kv # Prioritize explicit num_heads_kv
        if multi_query_attention and num_heads_kv is None: # Fallback to MQA flag if num_heads_kv not set
            actual_num_heads_kv_decoder = 1
        elif multi_query_attention and num_heads_kv is not None and num_heads_kv != 1:
             raise ValueError("multi_query_attention=True is incompatible with num_heads_kv != 1 unless num_heads_kv is None.")


        if with_encoder_attention and actual_num_heads_kv_decoder not in (None, 1, num_heads):
            if not (actual_num_heads_kv_decoder < num_heads and num_heads % actual_num_heads_kv_decoder == 0) :
                raise ValueError(
                    f"num_heads_kv={actual_num_heads_kv_decoder} is not supported in the cross-attention layers "
                    f"with num_heads={num_heads}. It must be None, 1, num_heads, or a divisor of num_heads for GQA."
                )

        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = pre_norm
        self.activation = np.dtype("int8").type(activation)
        self.alignment_layer = np.dtype("int16").type(alignment_layer)
        self.alignment_heads = np.dtype("int16").type(alignment_heads)
        self.embeddings = common_spec.EmbeddingsSpec()
        self.scale_embeddings = True
        self.scale_outputs = model_spec.OPTIONAL
        self.alibi = alibi
        self.alibi_use_positive_positions = alibi_use_positive_positions
        self.scale_alibi = scale_alibi

        if sliding_window is not None:
            self.sliding_window = np.dtype("int32").type(sliding_window)

        if not relative_position and not relative_attention_bias and not alibi and rotary_dim is None:
            self.position_encodings = PositionEncoderSpec()
        
        if pre_norm and not no_final_norm:
            self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        if layernorm_embedding:
            self.layernorm_embedding = common_spec.LayerNormSpec(rms_norm=rms_norm)
        
        self.projection = common_spec.LinearSpec() # For lm_head

        self.layer = [
            TransformerDecoderLayerSpec(
                with_encoder_attention=with_encoder_attention,
                relative_position=relative_position,
                relative_attention_bias=relative_attention_bias,
                ffn_glu=ffn_glu,
                rms_norm=rms_norm,
                # RoPE parameters for self-attention
                rotary_dim=rotary_dim,
                rotary_interleave=rotary_interleave,
                rotary_scaling_type=rotary_scaling_type,
                rotary_scaling_factor=rotary_scaling_factor,
                rotary_base=rotary_base,
                original_max_position_embeddings=original_max_position_embeddings,
                max_position_embeddings=max_position_embeddings,
                # Other structural params
                parallel_residual=parallel_residual,
                shared_layer_norm=shared_layer_norm,
                pre_post_layer_norm=pre_post_layer_norm,
                num_heads_kv=actual_num_heads_kv_decoder, # Pass effective num_heads_kv
                head_dim=head_dim, # Pass head_dim
                sliding_window=sliding_window,
            )
            for _ in range(num_layers)
        ]
        self.start_from_zero_embedding = False # Default

        # Store resolved MQA/GQA status for config if needed
        self._config["multi_query_attention"] = (actual_num_heads_kv_decoder == 1) or \
                                             (actual_num_heads_kv_decoder is not None and actual_num_heads_kv_decoder != num_heads)


        if project_in_out:
            self.project_in = common_spec.LinearSpec()
            self.project_out = common_spec.LinearSpec()

        if quant_type is not None:
            self._config["quantization_type"] = quant_type
            self._config["quantization_bits"] = quant_bits
            if quant_group_size is not None: # group_size can be optional for some quant methods
                self._config["quantization_group_size"] = quant_group_size
    
    @property
    def config(self): # Ensure this property exists
        return self._config


class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(
        self,
        relative_position: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        num_heads_kv: Optional[int] = None, # GQA/MQA param for self-attention
        sliding_window: Optional[int] = None, # Keep if used by MHA
        # RoPE arguments
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1.0,
        rotary_base: float = 10000.0,
        original_max_position_embeddings: int = 0,
        max_position_embeddings: int = 0,
        head_dim: Optional[int] = None, # If MHA spec needs it directly
    ):
        self.self_attention = attention_spec.MultiHeadAttentionSpec(
            self_attention=True, # Indicates this is a self-attention block
            relative_position=relative_position,
            relative_attention_bias=relative_attention_bias,
            rms_norm=rms_norm,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim, # Pass if MHA uses it
            sliding_window=sliding_window,
            # RoPE parameters
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,
        )
        self.ffn = FeedForwardSpec(glu=ffn_glu, rms_norm=rms_norm)


class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    def __init__(
        self,
        with_encoder_attention: bool = True,
        relative_position: bool = False, 
        relative_attention_bias: bool = False, 
        ffn_glu: bool = False,
        rms_norm: bool = False,
        # RoPE arguments (only for self-attention part)
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1.0,
        rotary_base: float = 10000.0,
        original_max_position_embeddings: int = 0,
        max_position_embeddings: int = 0,
        # Other structural args
        parallel_residual: bool = False,
        shared_layer_norm: bool = False,
        pre_post_layer_norm: bool = False,

        num_heads_kv: Optional[int] = None, 
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None, # For self-attention with sliding window
    ):
        # Self-attention part of the decoder layer (uses RoPE)
        self.self_attention = attention_spec.MultiHeadAttentionSpec(
            self_attention=True,
            relative_position=relative_position, 
            relative_attention_bias=relative_attention_bias,
            rms_norm=rms_norm,
            # RoPE parameters
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            sliding_window=sliding_window,
        )

        if with_encoder_attention:
            self.attention = attention_spec.MultiHeadAttentionSpec(
                self_attention=True, # Explicitly False, or rely on default
                rms_norm=rms_norm,
                num_heads_kv=num_heads_kv, 
                head_dim=head_dim,
            )
        
        self.ffn = FeedForwardSpec(glu=ffn_glu, rms_norm=rms_norm)

        # Handle parallel residual logic (GPT-J style)
        if parallel_residual:
            if shared_layer_norm: # Typically used if MQA (num_heads_kv=1) for GPT-J like models
                self.shared_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # One norm before both attn and FFN
            else: # Separate norms before attention and FFN
                self.input_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Before attention
                self.post_attention_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Before FFN

            # In parallel setups, the layer_norm within self_attention and ffn is usually removed
            # as the normalization is applied to the main residual stream.
            if hasattr(self.self_attention, "layer_norm"):
                delattr(self.self_attention, "layer_norm")
            if hasattr(self.ffn, "layer_norm"):
                delattr(self.ffn, "layer_norm")
        

        if pre_post_layer_norm:
            # These are the "pre" norms for the sub-layers
            self.input_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Pre-SelfAttention
            self.pre_feedforward_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Pre-FFN
            

            if with_encoder_attention:
                self.post_attention_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Post-SelfAttention (or Post-CrossAttention if order differs)
            else: # If no cross-attention, this might be just post self-attention
                 self.post_attention_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Post-SelfAttention

            self.post_feedforward_layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm) # Post-FFN

            # If pre_post_layer_norm is true, the individual layer_norms within MHA and FFN are often disabled
            if hasattr(self.self_attention, "layer_norm"):
                delattr(self.self_attention, "layer_norm")
            if hasattr(self.attention, "layer_norm") and with_encoder_attention: # For cross-attention
                 delattr(self.attention, "layer_norm")
            if hasattr(self.ffn, "layer_norm"):
                delattr(self.ffn, "layer_norm")


class FeedForwardSpec(model_spec.LayerSpec):
    def __init__(self, glu=False, rms_norm=False):
        self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)
        self.linear_0 = common_spec.LinearSpec()
        self.linear_1 = common_spec.LinearSpec()
        if glu:
            self.linear_0_noact = common_spec.LinearSpec()


class PositionEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL


class TransformerConfig(model_spec.SequenceToSequenceModelConfig):
    """Configuration for Transformer models."""

    def __init__(self, layer_norm_epsilon: Optional[float] = None, **kwargs):
        """Initializes the configuration for Transformer models.

        Args:
          layer_norm_epsilon: The layer norm epsilon value.
          **kwargs: Additional configuration.
        """
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class TransformerSpec(model_spec.SequenceToSequenceModelSpec):
    def __init__(
        self,
        encoder: TransformerEncoderSpec,
        decoder: TransformerDecoderSpec,
    ):
        if not isinstance(encoder, TransformerEncoderSpec):
            raise TypeError("encoder argument must be a TransformerEncoderSpec")
        if not isinstance(decoder, TransformerDecoderSpec):
            raise TypeError("decoder argument must be a TransformerDecoderSpec")

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Make sure _config exists if you're adding attributes like this
        if not hasattr(self, "_config"):
            self._config = model_spec.ModelConfig() # Or however ModelConfig is usually initialized
        self._config.add_attribute("multi_query_attention", False) # Example

    @classmethod
    def from_config(
        cls,
        num_layers: Union[int, Tuple[int, int]],
        num_heads: int,
        with_relative_position: bool = False, # Keep existing args
        pre_norm: bool = True,
        no_final_norm: bool = False,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        alignment_layer: int = -1,
        alignment_heads: int = 1,
        num_source_embeddings: int = 1,
        embeddings_merge: common_spec.EmbeddingsMerge = common_spec.EmbeddingsMerge.CONCAT,
        layernorm_embedding: bool = False,
        relative_attention_bias: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        multi_query_attention: bool = False,
        # ADDED RoPE ARGUMENTS HERE (must match what your loader sends)
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1.0,
        rotary_base: float = 10000.0,
        original_max_position_embeddings: int = 0,
        max_position_embeddings: int = 0,

    ):
        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers

        encoder = TransformerEncoderSpec(
            num_encoder_layers,
            num_heads,
            pre_norm=pre_norm,
            no_final_norm=no_final_norm,
            activation=activation,
            num_source_embeddings=num_source_embeddings,
            embeddings_merge=embeddings_merge,
            layernorm_embedding=layernorm_embedding,
            relative_position=with_relative_position, # name was with_relative_position
            relative_attention_bias=relative_attention_bias,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
            multi_query_attention=multi_query_attention,
            # Pass RoPE args to TransformerEncoderSpec
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,
            # num_heads_kv = num_heads_kv, # If applicable for encoder
            # head_dim = head_dim, # If applicable for encoder
        )

        decoder = TransformerDecoderSpec(
            num_decoder_layers,
            num_heads,
            pre_norm=pre_norm,
            activation=activation, # Corrected from no_final_norm
            layernorm_embedding=layernorm_embedding,
            # with_encoder_attention=True, # Default, ensure it's handled if configurable
            no_final_norm=no_final_norm, # This was missing from your original snippet for decoder
            # project_in_out=False, # Default, ensure it's handled if configurable
            relative_position=with_relative_position, # name was with_relative_position
            relative_attention_bias=relative_attention_bias,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
            # alibi=False, # Default
            # alibi_use_positive_positions=False, # Default
            # scale_alibi=False, # Default
            # Pass RoPE args to TransformerDecoderSpec
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,
            # parallel_residual=False, # Default
            # shared_layer_norm=False, # Default
            # pre_post_layer_norm=False, # Default
            multi_query_attention=multi_query_attention, # Pass this through
            # num_heads_kv=num_heads_kv, # If applicable for decoder
            # head_dim=head_dim, # If applicable for decoder
            # sliding_window=None, # Default
        )

        return cls(encoder, decoder)

    @property
    def name(self):
        return "TransformerSpec"

    @property
    def revision(self):
        return 7

    def get_default_config(self):
        return TransformerConfig()

    def get_source_vocabulary_size(self):
        return [spec.weight.shape[0] for spec in self.encoder.embeddings]

    def get_target_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class TransformerDecoderModelConfig(model_spec.LanguageModelConfig):
    """Configuration for Transformer decoder models."""

    def __init__(self, layer_norm_epsilon: Optional[float] = None, **kwargs):
        """Initializes the configuration for Transformer decoder models.

        Args:
          layer_norm_epsilon: The layer norm epsilon value.
          **kwargs: Additional configuration.
        """
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class TransformerDecoderModelSpec(model_spec.LanguageModelSpec):
    """Describes a Transformer decoder model (e.g. GPT-2)."""

    def __init__(self, decoder: TransformerDecoderSpec):
        """Initializes a Transformer decoder model specification.

        Args:
          decoder: The decoder specification.
        """
        if not isinstance(decoder, TransformerDecoderSpec):
            raise TypeError("decoder argument must be a TransformerDecoderSpec")

        super().__init__()
        self.decoder = decoder
        for key, value in self.decoder.config.items():
            self._config.add_attribute(key, value)

    @classmethod
    def from_config(
        cls,
        num_layers: int,
        num_heads: int,
        pre_norm: bool = True,
        activation: common_spec.Activation = common_spec.Activation.RELU,
        layernorm_embedding: bool = False,
        no_final_norm: bool = False,
        project_in_out: bool = False,
        with_relative_position: bool = False,
        ffn_glu: bool = False,
        rms_norm: bool = False,
        alibi: bool = False,
        alibi_use_positive_positions: bool = False,
        scale_alibi: bool = False,
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = True,
        rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = None,
        rotary_scaling_factor: float = 1,
        rotary_base: float = 10000,
        original_max_position_embeddings: int = 0,
        max_position_embeddings: int = 0,
        parallel_residual: bool = False,
        shared_layer_norm: bool = False,
        pre_post_layer_norm: bool = False,
        multi_query_attention: bool = False,
        num_heads_kv: Optional[int] = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None,
        quant_type: Optional[common_spec.Quantization] = None,
        quant_group_size: Optional[int] = None,
        quant_bits: Optional[int] = None,
    ):
        """Creates a Transformer decoder model specification.

        Args:
          num_layers: Number of decoder layers.
          num_heads: Number of attention heads.
          pre_norm: Enable the pre-norm Transformer architecture.
          activation: Activation to apply in the feed-forward network.
          layernorm_embedding: Apply layer normalization after the embedding layer.
          no_final_norm: Do not apply layer normalization after the last decoder block.
          project_in_out: Add a linear layer after the embedding layer and another one
            before the final output projection.
          with_relative_position: Enable relative position representations modules.
          ffn_glu: Use gated linear units in the FFN layers as described in
            https://arxiv.org/abs/2002.05202.
          rms_norm: Use the root mean square layer normalization.
          alibi: Use attention with linear biases.
          alibi_use_positive_positions: Use positive positions in the ALiBi definition.
          scale_alibi: Apply the dot product scale factor to ALiBi.
          rotary_dim: Apply rotary embeddings to these first N dimensions. If 0, rotary
            embeddings are applied to all dimensions.
          rotary_interleave: Interleave the head dimensions when rotary embeddings are applied.
            Otherwise the head dimensions are sliced in half.
          rotary_scaling_type: Type of RoPE scaling.
          rotary_scaling_factor: Factor used in the RoPE scaling.
          rotary_base: The base period of the rotary embeddings.
          original_max_position_embeddings: The original max position embeddings
            for Su rope embeddings
          max_position_embeddings: The max position embeddings for Su rope embeddings
          parallel_residual: Use parallel residual connections in each layer block, as used
            by the GPT-J and GPT-NeoX models.
          shared_layer_norm: When using parallel residual, share the input and post
            attention layer norms.
          pre_post_layer_norm: add post layer norm for each pre norm layer
          multi_query_attention: Use multi-query attention (alias for num_heads_kv=1).
          num_heads_kv: Number of attention heads for the key and value.
          head_dim: Number of head
          sliding_window: max sequence length to retain KV cache
          quant_type: quantization type used (like awq... for lower bit quantization)
          quant_group_size: group size of the lower bit quantization
          quant_bits: number of bit of the quantization (ex: 4bit)
        """
        decoder = TransformerDecoderSpec(
            num_layers,
            num_heads,
            pre_norm=pre_norm,
            activation=activation,
            layernorm_embedding=layernorm_embedding,
            with_encoder_attention=False,
            no_final_norm=no_final_norm,
            project_in_out=project_in_out,
            relative_position=with_relative_position,
            ffn_glu=ffn_glu,
            rms_norm=rms_norm,
            alibi=alibi,
            alibi_use_positive_positions=alibi_use_positive_positions,
            scale_alibi=scale_alibi,
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_scaling_type=rotary_scaling_type,
            rotary_scaling_factor=rotary_scaling_factor,
            rotary_base=rotary_base,
            original_max_position_embeddings=original_max_position_embeddings,
            max_position_embeddings=max_position_embeddings,
            parallel_residual=parallel_residual,
            shared_layer_norm=shared_layer_norm,
            pre_post_layer_norm=pre_post_layer_norm,
            multi_query_attention=multi_query_attention,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            sliding_window=sliding_window,
            quant_type=quant_type,
            quant_group_size=quant_group_size,
            quant_bits=quant_bits,
        )

        return cls(decoder)

    @property
    def name(self):
        return "TransformerDecoderSpec"

    @property
    def revision(self):
        return 8

    def get_default_config(self):
        return TransformerDecoderModelConfig()

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]


class TransformerEncoderModelConfig(model_spec.LanguageModelConfig):
    """Configuration for Transformer encoder models."""

    def __init__(self, layer_norm_epsilon: Optional[float] = None, **kwargs):
        """Initializes the configuration for Transformer encoder models.

        Args:
          layer_norm_epsilon: The layer norm epsilon value.
          **kwargs: Additional configuration.
        """
        super().__init__(layer_norm_epsilon=layer_norm_epsilon, **kwargs)


class TransformerEncoderModelSpec(model_spec.LanguageModelSpec):
    """Describes a Transformer encoder model (e.g. BERT)."""

    def __init__(
        self,
        encoder: TransformerEncoderSpec,
        pooling_layer: bool = False,
        pooling_activation: common_spec.Activation = common_spec.Activation.Tanh,
    ):
        """Initializes a Transformer encoder model specification.

        Args:
          encoder: The encoder specification.
          pooling_layer: Add the pooling layer.
          pooling_activation: The activation to apply after the pooling layer.
        """
        if not isinstance(encoder, TransformerEncoderSpec):
            raise TypeError("encoder argument must be a TransformerEncoderSpec")

        super().__init__()
        self.encoder = encoder
        self._config.add_attribute(
            "multi_query_attention", self.encoder.multi_query_attention
        )

        if pooling_layer:
            self.pooler_dense = common_spec.LinearSpec()
            self.pooler_activation = np.dtype("int8").type(pooling_activation)

    @property
    def name(self):
        return "TransformerEncoderSpec"

    @property
    def revision(self):
        return 1

    def get_default_config(self):
        return TransformerEncoderModelConfig()

    def get_vocabulary_size(self):
        return self.encoder.embeddings[0].weight.shape[0]
