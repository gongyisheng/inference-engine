# Phase 0 — "Hello, one token": Qwen3 from-scratch forward + greedy decode

**Date:** 2026-06-19
**Status:** Approved design
**Roadmap phase:** 0 (see `docs/roadmap.md`)

## Goal

Load Qwen3-4B with raw tensor operations (no `transformers` modeling code), run a
naive forward pass (no KV cache), greedy-decode tokens, and prove **token-level
parity** with HF `transformers` as the golden reference. This test is re-run by
every later phase to catch correctness regressions.

## Scope

**In scope**
- Load weights from Hugging Face (`huggingface_hub` + `safetensors`).
- From-scratch model: embedding, Qwen3 transformer blocks, final norm, LM head.
- Naive greedy generation: full-sequence forward every step, no cache.
- Golden-reference accuracy test vs. HF `transformers`.

**Out of scope (later phases)**
- KV cache (Phase 1), batching (Phase 2), PagedAttention (Phase 3),
  continuous batching (Phase 4), prefix caching (Phase 5),
  sampling temp/top-k/top-p + stopping (Phase 6), CUDA graphs/kernels (Phase 7),
  API server (Phase 8), MoE (Phase 9), tensor parallelism (Phase 10).

## Target

- **Model:** Qwen3-4B (dense). Config driven, so other dense sizes work too.
- **Hardware:** single GPU, `cuda:0`, dtype bf16. ~8 GB weights, fits 16 GB.

## Module layout

All under `src/inference_engine/`, mirroring the reference repo
(<https://github.com/gongyisheng/llm-from-scratch/tree/main/qwen3>).

| Module | Responsibility |
|---|---|
| `config.py` | `Qwen3Config` dataclass loaded from HF `config.json`: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, `max_position_embeddings`. |
| `tokenizer.py` | Thin wrapper over HF `AutoTokenizer`; expose `encode`, `decode`, and `apply_chat_template`. |
| `layers.py` | `RMSNorm`; RoPE precompute + apply; GQA attention with **per-head QK-norm**; SwiGLU `MLP`; `TransformerBlock` (pre-norm). |
| `model.py` | `Qwen3Model`: embed → N `TransformerBlock` → final `RMSNorm` → LM head (tied to embedding weight when `tie_word_embeddings`). |
| `weights.py` | Download snapshot, open `safetensors`, map HF parameter names → our module state dict; load into model. |
| `generate.py` | `greedy_generate(model, input_ids, max_new_tokens)`: loop forward over the full sequence, take `argmax` of last-position logits, append, stop on EOS or `max_new_tokens`. |
| `main.py` | CLI: takes a prompt, applies chat template, generates, prints text. |

## Data flow

```
token_ids
  → embedding lookup
  → for each of num_hidden_layers blocks:
        h = x + GQA(RMSNorm(x))        # attention sublayer, pre-norm + residual
        x = h + SwiGLU(RMSNorm(h))     # FFN sublayer, pre-norm + residual
  → final RMSNorm
  → LM head (linear; weight tied to embedding when configured)
  → logits[..., -1, :]
  → argmax → next token
```

### GQA attention block (per layer)
1. Project x → q, k, v with `head_dim`-sized heads
   (`num_attention_heads` for q, `num_key_value_heads` for k/v).
2. **QK-norm:** apply `RMSNorm` over each head's `head_dim` to q and k.
3. Apply RoPE (`rope_theta`) to q and k.
4. Repeat k/v heads to match q heads (GQA grouping).
5. Causal scaled-dot-product attention (use `torch.nn.functional.scaled_dot_product_attention`, `is_causal=True`).
6. Output projection.

## Qwen3-specific details to verify

- **GQA:** 4B uses 32 query heads / 8 KV heads.
- **QK-norm:** RMSNorm applied per head to q and k *before* RoPE — a Qwen3 hallmark, easy to miss.
- **Explicit `head_dim`:** read from config; may not equal `hidden_size / num_attention_heads`.
- **SwiGLU:** `down(silu(gate(x)) * up(x))`.
- **Tied embeddings:** LM head shares the embedding weight when `tie_word_embeddings` is true.
- **No attention bias** (Qwen3 attention/MLP linears have no bias); confirm against config.

## Verification (non-negotiable golden reference)

`tests/test_accuracy.py`:
- Load the same Qwen3-4B model with HF `transformers` (reference) and with our engine.
- For a few fixed prompts:
  - Compare last-position logits: assert max absolute difference under a tolerance
    (bf16, so a small tolerance, e.g. < 0.5 on raw logits; refine empirically).
  - Assert **identical greedy token sequences** for N new tokens.
- Run on `cuda:0`, bf16, deterministic (greedy → no RNG).

A lightweight `tests/test_knowledge.py` (sanity: model answers a trivial factual
prompt) is optional but useful as a smoke test.

## Success criteria

1. `python -m inference_engine.main --prompt "..."` produces coherent text.
2. `pytest tests/test_accuracy.py` passes: greedy tokens match HF transformers.
3. Code uses no `transformers` *modeling* classes (tokenizer + reference-only use allowed).
