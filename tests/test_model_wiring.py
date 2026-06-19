import torch

from inference_engine.config import Qwen3Config
from inference_engine.model import Qwen3ForCausalLM


def tiny_cfg():
    return Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=50, rope_theta=1e6, rms_norm_eps=1e-6,
        tie_word_embeddings=True, max_position_embeddings=128, eos_token_id=0,
    )


def test_forward_shape():
    cfg = tiny_cfg()
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (1, 7, cfg.vocab_size)


def test_statedict_keys_match_hf_names():
    cfg = tiny_cfg()
    keys = set(Qwen3ForCausalLM(cfg).state_dict().keys())
    assert "model.embed_tokens.weight" in keys
    assert "model.layers.0.self_attn.q_norm.weight" in keys
    assert "model.norm.weight" in keys
    assert "lm_head.weight" in keys
