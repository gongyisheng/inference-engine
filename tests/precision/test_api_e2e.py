import pytest
import torch
from fastapi.testclient import TestClient

from inference_engine.api.server import create_app

torch_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@torch_cuda
def test_end_to_end_real_model():
    from inference_engine.engine import LLM

    llm = LLM("Qwen/Qwen3-4B", device="cuda:0")
    c = TestClient(create_app(llm))
    r = c.post("/v1/chat/completions", json={
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["choices"][0]["message"]["content"]) > 0
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
