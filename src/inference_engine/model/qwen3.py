import torch
import torch.nn as nn

from ..layers.block import TransformerBlock
from ..layers.pos_emb import rope_cos_sin
from ..utils.config import Qwen3Config
from .transformer import Transformer


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = Transformer(
            cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.rms_norm_eps,
            block_factory=lambda: TransformerBlock(cfg),
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, cache=None):
        _, s = input_ids.shape
        past = cache.length if cache is not None else 0
        dtype = self.model.embed_tokens.weight.dtype
        cos, sin = rope_cos_sin(s, self.cfg.head_dim, self.cfg.rope_theta, input_ids.device, dtype, offset=past)
        h = self.model(input_ids, cos, sin, cache)
        return self.lm_head(h)
