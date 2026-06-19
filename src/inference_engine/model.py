import torch
import torch.nn as nn

from .config import Qwen3Config
from .layers import RMSNorm, TransformerBlock


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, cos, sin):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids):
        _, s = input_ids.shape
        device = input_ids.device
        hd = self.cfg.head_dim
        inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        pos = torch.arange(s, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = self.model.embed_tokens.weight.dtype
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        h = self.model(input_ids, cos, sin)
        return self.lm_head(h)
