import json
from dataclasses import dataclass


@dataclass
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    eos_token_id: int | list[int]

    @classmethod
    def from_json(cls, path: str) -> "Qwen3Config":
        with open(path) as f:
            c = json.load(f)
        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            vocab_size=c["vocab_size"],
            rope_theta=c["rope_theta"],
            rms_norm_eps=c["rms_norm_eps"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            max_position_embeddings=c["max_position_embeddings"],
            eos_token_id=c["eos_token_id"],
        )
