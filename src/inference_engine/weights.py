import glob
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .config import Qwen3Config
from .model import Qwen3ForCausalLM


def load_model(model_id: str = "Qwen/Qwen3-4B", device: str = "cuda:0", dtype=torch.bfloat16):
    path = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    cfg = Qwen3Config.from_json(os.path.join(path, "config.json"))

    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)

    state = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        state.update(load_file(f))
    if cfg.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg, path
