import torch

from inference_engine.config import Qwen3Config
from inference_engine.model import Qwen3ForCausalLM
from inference_engine.generate import greedy_generate


def test_greedy_appends_tokens():
    cfg = Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=50, rope_theta=1e6, rms_norm_eps=1e-6,
        tie_word_embeddings=True, max_position_embeddings=128, eos_token_id=-1,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    out = greedy_generate(model, ids, max_new_tokens=5, eos_token_id=cfg.eos_token_id)
    assert out.shape == (1, 8)
    assert torch.equal(out[:, :3], ids)
