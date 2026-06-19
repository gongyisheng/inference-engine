import torch

from inference_engine.cache import KVCache
from inference_engine.model.qwen3 import Qwen3ForCausalLM
from inference_engine.utils.config import Qwen3Config


def _tiny_model():
    cfg = Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=50, rope_theta=1e6, rms_norm_eps=1e-6,
        tie_word_embeddings=True, max_position_embeddings=128, eos_token_id=-1,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


@torch.inference_mode()
def test_cached_decode_matches_full_recompute():
    model = _tiny_model()
    ids = torch.randint(0, 50, (1, 6))

    full = model(ids)  # no cache: logits for every position in one pass

    cache = KVCache(model.cfg.num_hidden_layers)
    step_logits = []
    for t in range(ids.shape[1]):
        out = model(ids[:, t:t + 1], cache)  # one token at a time
        step_logits.append(out[:, -1, :])
    cached = torch.stack(step_logits, dim=1)

    assert torch.allclose(full, cached, atol=1e-4, rtol=0.0)
