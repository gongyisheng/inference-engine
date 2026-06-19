import json
from inference_engine.config import Qwen3Config


def test_from_json(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True, "max_position_embeddings": 40960,
        "eos_token_id": 151645,
    }))
    cfg = Qwen3Config.from_json(str(cfg_path))
    assert cfg.head_dim == 128
    assert cfg.num_key_value_heads == 8
    assert cfg.tie_word_embeddings is True


def test_head_dim_defaults(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "hidden_size": 2048, "intermediate_size": 4096, "num_hidden_layers": 2,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "vocab_size": 100, "rope_theta": 1e6, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096, "eos_token_id": 0,
    }))
    cfg = Qwen3Config.from_json(str(cfg_path))
    assert cfg.head_dim == 128  # 2048 / 16
    assert cfg.tie_word_embeddings is False  # default
