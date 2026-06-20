import torch.nn as nn

from ..utils.config import Qwen3Config
from .attention import Attention
from .mlp import MLP
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
