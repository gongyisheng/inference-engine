"""Local engine micro-benchmark (no HTTP).

Measures prefill (first forward over the prompt) vs decode (each subsequent
single-token forward) separately. Decode feeds one token at a time through a
KVCache, so each step is O(n) instead of O(n^2) recompute.
"""
import argparse
import time

import torch

from inference_engine.cache import KVCache
from inference_engine.engine import LLM


@torch.inference_mode()
def run(llm: LLM, prompt_len: int, output_len: int):
    device = llm.device
    vocab = llm.cfg.vocab_size
    ids = torch.randint(0, vocab, (1, prompt_len), device=device)

    def sync():
        if "cuda" in str(device):
            torch.cuda.synchronize()

    cache = KVCache(llm.cfg.num_hidden_layers)

    sync()
    t0 = time.perf_counter()
    logits = llm.model(ids, cache)
    sync()
    prefill_s = time.perf_counter() - t0

    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync()
    t0 = time.perf_counter()
    for _ in range(output_len - 1):
        logits = llm.model(next_id, cache)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync()
    decode_s = time.perf_counter() - t0

    decode_tokens = max(output_len - 1, 0)
    print(f"prompt_len={prompt_len} output_len={output_len}")
    print(f"prefill: {prompt_len} tok in {prefill_s*1e3:.1f} ms -> {prompt_len/prefill_s:.1f} tok/s")
    if decode_tokens:
        print(f"decode:  {decode_tokens} tok in {decode_s*1e3:.1f} ms -> {decode_tokens/decode_s:.1f} tok/s")
    total_s = prefill_s + decode_s
    print(f"total:   {output_len} new tok in {total_s*1e3:.1f} ms -> {output_len/total_s:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    args = parser.parse_args()

    llm = LLM(args.model, device=args.device)
    run(llm, args.prompt_len, args.output_len)


if __name__ == "__main__":
    main()
