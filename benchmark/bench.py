"""Local engine micro-benchmark (no HTTP).

Measures prefill (first forward over the prompt) vs decode (each subsequent
single-token forward) separately. Decode feeds one token at a time through a
KVCache, so each step is O(n) instead of O(n^2) recompute.

NVTX ranges ("prefill", "decode", "decode_step") annotate the timeline. With
--profile the script warms up, then brackets the measured region with
cudaProfilerStart/Stop so Nsight Systems captures only steady state:

    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \\
        -o bench_report --force-overwrite true \\
        uv run python benchmark/bench.py --profile --prompt-len 512 --output-len 64
    nsys stats bench_report.nsys-rep
"""
import argparse
import contextlib
import time

import torch

from inference_engine.cache import KVCache
from inference_engine.engine import LLM


def _nvtx(msg: str, enabled: bool):
    return torch.cuda.nvtx.range(msg) if enabled else contextlib.nullcontext()


@torch.inference_mode()
def _decode(llm: LLM, next_id, cache, steps: int, nvtx: bool):
    for _ in range(steps):
        with _nvtx("decode_step", nvtx):
            logits = llm.model(next_id, cache)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_id


@torch.inference_mode()
def run(llm: LLM, prompt_len: int, output_len: int, profile: bool = False):
    device = llm.device
    is_cuda = "cuda" in str(device)
    ids = torch.randint(0, llm.cfg.vocab_size, (1, prompt_len), device=device)

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    if profile and is_cuda:  # warm up kernels so the capture is steady-state
        warm = KVCache(llm.cfg.num_hidden_layers)
        nid = llm.model(ids, warm)[:, -1, :].argmax(dim=-1, keepdim=True)
        _decode(llm, nid, warm, min(8, max(output_len - 1, 0)), is_cuda)
        sync()
        torch.cuda.profiler.start()

    cache = KVCache(llm.cfg.num_hidden_layers)

    sync()
    t0 = time.perf_counter()
    with _nvtx("prefill", is_cuda):
        logits = llm.model(ids, cache)
    sync()
    prefill_s = time.perf_counter() - t0

    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync()
    t0 = time.perf_counter()
    with _nvtx("decode", is_cuda):
        _decode(llm, next_id, cache, output_len - 1, is_cuda)
    sync()
    decode_s = time.perf_counter() - t0

    if profile and is_cuda:
        torch.cuda.profiler.stop()

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
    parser.add_argument("--profile", action="store_true",
                        help="warm up and bracket the measured region with cudaProfilerStart/Stop for nsys")
    args = parser.parse_args()

    llm = LLM(args.model, device=args.device)
    run(llm, args.prompt_len, args.output_len, profile=args.profile)


if __name__ == "__main__":
    main()
