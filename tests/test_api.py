import pytest
import torch
from fastapi.testclient import TestClient

from inference_engine.api.server import create_app

torch_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class FakeEngine:
    def __init__(self):
        self.last_call = None

    def generate(self, messages, max_tokens):
        self.last_call = (messages, max_tokens)
        return {"text": "hello there", "prompt_tokens": 3,
                "completion_tokens": 2, "finish_reason": "stop"}


def client(engine=None, **kw):
    return TestClient(create_app(engine or FakeEngine(), **kw))


def test_chat_completion_ok():
    c = client()
    r = c.post("/v1/chat/completions", json={
        "model": "qwen", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello there"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_messages_forwarded_with_max_tokens():
    eng = FakeEngine()
    client(eng).post("/v1/chat/completions", json={
        "model": "qwen", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
    })
    assert eng.last_call == ([{"role": "user", "content": "hi"}], 8)


def test_default_max_tokens_used_when_omitted():
    eng = FakeEngine()
    client(eng, default_max_tokens=99).post("/v1/chat/completions", json={
        "model": "qwen", "messages": [{"role": "user", "content": "hi"}],
    })
    assert eng.last_call[1] == 99


def test_empty_messages_returns_400():
    c = client()
    r = c.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
    assert r.status_code == 400
    assert "error" in r.json()


def test_max_tokens_zero_is_respected():
    eng = FakeEngine()
    client(eng).post("/v1/chat/completions", json={
        "model": "qwen", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0,
    })
    assert eng.last_call[1] == 0


def test_missing_field_returns_422():
    c = client()
    r = c.post("/v1/chat/completions", json={"model": "qwen"})
    assert r.status_code == 422


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
