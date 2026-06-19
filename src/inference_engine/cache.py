import torch


class KVCache:
    """Per-layer key/value store that grows one decode step at a time.

    Stores K/V with the model's `num_key_value_heads` (before GQA expansion):
    shape [b, n_kv, seq, head_dim]. Growth is by concatenation — O(n) copy per
    step, but turns decode from O(n^2) recompute into O(n). Pre-allocation is a
    later phase.
    """

    def __init__(self, num_layers: int):
        self.entries: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    @property
    def length(self) -> int:
        e = self.entries[0]
        return 0 if e is None else e[0].shape[2]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        prev = self.entries[layer_idx]
        if prev is not None:
            k = torch.cat([prev[0], k], dim=2)
            v = torch.cat([prev[1], v], dim=2)
        self.entries[layer_idx] = (k, v)
        return k, v
