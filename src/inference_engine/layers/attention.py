import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import Qwen3Config
from .norm import RMSNorm
from .pos_emb import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None, layer_idx=0):
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, s, self.n_kv, self.head_dim))
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        k_len = k.shape[2]
        if s == k_len:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # prefill
        elif s == 1:
            o = F.scaled_dot_product_attention(q, k, v)  # decode: lone query sees all keys
        else:
            # chunked decode: query at offset (k_len - s + i) attends to keys 0..that.
            q_pos = torch.arange(k_len - s, k_len, device=q.device).unsqueeze(1)
            k_pos = torch.arange(k_len, device=q.device).unsqueeze(0)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=k_pos <= q_pos)
        o = o.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(o)
