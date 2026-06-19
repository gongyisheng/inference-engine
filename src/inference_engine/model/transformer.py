from typing import Callable

import torch.nn as nn

from ..layers.norm import RMSNorm


class Transformer(nn.Module):
    """Generic decoder stack: embed -> blocks -> final norm.

    `block_factory` builds one block; each block takes `(x, cos, sin)`.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, eps: float,
                 block_factory: Callable[[], nn.Module]):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([block_factory() for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, input_ids, cos, sin):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)
