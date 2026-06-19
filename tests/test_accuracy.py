import pytest
import torch

MODEL_ID = "Qwen/Qwen3-4B"
PROMPTS = ["The capital of France is", "Once upon a time", "2 + 2 ="]

torch_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@torch_cuda
def test_greedy_tokens_match_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from inference_engine.weights import load_model
    from inference_engine.generate import greedy_generate

    ours, cfg, path = load_model(MODEL_ID, device="cuda:0")
    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda:1").eval()
    tok = AutoTokenizer.from_pretrained(path)

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids
        our_out = greedy_generate(ours, ids.to("cuda:0"), max_new_tokens=20, eos_token_id=cfg.eos_token_id)
        our_new = our_out[0, ids.shape[1]:].tolist()
        with torch.inference_mode():
            ref_out = ref.generate(ids.to("cuda:1"), max_new_tokens=20, do_sample=False)
        ref_new = ref_out[0, ids.shape[1]:].tolist()
        assert our_new == ref_new[:len(our_new)], f"mismatch for {prompt!r}: {our_new} != {ref_new}"


@torch_cuda
def test_last_logits_argmax_match_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from inference_engine.weights import load_model

    ours, cfg, path = load_model(MODEL_ID, device="cuda:0")
    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda:1").eval()
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids
    with torch.inference_mode():
        our_logits = ours(ids.to("cuda:0"))[0, -1].float().cpu()
        ref_logits = ref(ids.to("cuda:1")).logits[0, -1].float().cpu()
    assert our_logits.argmax().item() == ref_logits.argmax().item()
    assert torch.allclose(our_logits, ref_logits, atol=1.0, rtol=0.0)
