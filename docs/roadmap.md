# Inference Engine — Roadmap

A learning project: build an LLM inference engine in the spirit of **nano-vllm** / **mini-sglang**.
We build incrementally (vertical slices) — every phase is a working, testable engine.

## Targets

- **Hardware:** single NVIDIA GPU first; multi-GPU (tensor parallelism) later.
- **Models:** Qwen3 dense → Qwen3-MoE.
- **Kernels:** use FlashAttention etc.; the *engine architecture* is the learning focus. Perf passes come later.
- **Milestones:** **A** library (`LLM.generate`) → **B** OpenAI-compatible server → **C** full service.

## Target Architecture

```
Entry point: LLM (library) / API Server (B,C)
        │
   LLMEngine — orchestrator, the step() heartbeat
     ├── Scheduler        waiting/running queues, continuous batching, preemption
     │     └── BlockManager   PagedAttention: KV blocks, alloc/free, ref-count + prefix cache
     └── ModelRunner      holds model+weights, builds batch, runs forward, samples
           └── Model (Qwen3 / Qwen3-MoE)   RMSNorm, RoPE, GQA, SwiGLU, MoE FFN; paged-attn reads

Cross-cutting: Sequence (per-request state) · SamplingParams · Config
```

**The two defining ideas:**
1. **PagedAttention** — KV cache stored in fixed-size blocks (like OS pages); no fragmentation, easy prefix sharing. Owned by `BlockManager`.
2. **Continuous batching** — the batch is re-decided every forward pass; finished sequences leave and waiting ones join mid-flight. Lives in `LLMEngine.step()` / `Scheduler`.

## Phases

Pattern each phase: *feel a limitation → learn the fix → build it → verify against the golden reference.*

| # | Phase | Motivating limitation | What you build | Verify |
|---|-------|----------------------|----------------|--------|
| 0 | Hello, one token | — | Load Qwen3 weights, naive forward, greedy decode (no cache) | **Matches HF `transformers`** — the golden reference for all later phases |
| 1 | KV cache (contiguous) | O(n²) recompute is slow | Per-layer K/V cache; prefill vs decode | Same output, ~linear decode; measure tok/s |
| 2 | Static batching | One sequence wastes the GPU | Padded N-prompt batch + mask | Batched == per-sequence output |
| 3 | PagedAttention | Padding + contiguous cache wastes/fragments memory | `BlockManager`, block tables, FlashAttn paged/varlen path | Output matches; memory drops; no padding |
| 4 | Continuous batching | Static batch waits on slowest seq | `Scheduler` (waiting/running) + `LLMEngine.step()`; mix prefill/decode; preemption | Throughput beats static; output matches |
| 5 | Prefix caching | Shared prefixes recomputed per request | Content-hash blocks, ref-count, reuse | Shared system prompt → 2nd prefill near-free |
| 6 | Sampling + stopping | Greedy-only is a toy | temp / top-k / top-p, EOS / max-tokens / stop-strings, per-req `SamplingParams` | temp=0 deterministic; stops fire |
| 7 | **[perf — later]** CUDA graphs + kernels | Launch overhead / kernel speed | Capture decode as CUDA graph; optional Triton deep-dives | Same output, faster — *revisit after C* |
| 8 | **API server (B)** | Library can't serve | Async OpenAI-compatible server: `/v1/chat/completions`, SSE streaming, chat templates | Real streamed requests; OpenAI client shape |
| 9 | Qwen3-MoE | Dense FFN only | MoE block: router, expert selection, grouped GEMM | MoE output matches HF reference |
| 10 | Tensor parallelism | Single GPU | Weight sharding, NCCL all-reduce, worker processes | 2-GPU == 1-GPU output; throughput scales |
| 11 | **Service hardening (C)** | Engine ≠ service | Metrics (tput/latency/KV util), cancellation, abort | Load test; cancel frees blocks; metrics sane |

## Working agreement

- Claude teaches the concept + the *why*, sketches interfaces/data structures, and points at nano-vllm code to compare against. You write the implementation; we review and debug together.
- The **Phase 0 golden-reference test is non-negotiable** — every later phase re-runs it to catch correctness regressions immediately.
