from __future__ import annotations

from typing import Optional

from transformers import PretrainedConfig


class BabyLlamaKimiConfig(PretrainedConfig):
    model_type = "babyllama_kimi"

    def __init__(
        self,
        vocab_size: int = 64793,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        multiple_of: int = 32,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 512,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.multiple_of = multiple_of
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.rope_theta = rope_theta

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

