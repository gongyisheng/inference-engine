import torch

from .generate import greedy_generate
from .utils.tokenizer import Tokenizer
from .utils.weights import load_model


class LLM:
    """Stable facade over the engine. Greedy-only for now."""

    def __init__(self, model_id: str, device: str = "cuda:0", dtype=torch.bfloat16):
        self.model, self.cfg, path = load_model(model_id, device=device, dtype=dtype)
        self.tokenizer = Tokenizer(path)
        self.device = device

    def generate(self, messages: list[dict], max_tokens: int, enable_thinking: bool = True) -> dict:
        ids = self.tokenizer.apply_chat(messages, enable_thinking=enable_thinking).to(self.device)
        prompt_tokens = ids.shape[1]
        out = greedy_generate(self.model, ids, max_tokens, self.cfg.eos_token_id)
        new_ids = out[0, prompt_tokens:]
        completion_tokens = int(new_ids.shape[0])
        eos_set = self.cfg.eos_token_id
        eos_set = eos_set if isinstance(eos_set, list) else [eos_set]
        hit_eos = completion_tokens > 0 and int(new_ids[-1].item()) in eos_set
        return {
            "text": self.tokenizer.decode(new_ids),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "stop" if hit_eos else "length",
        }
