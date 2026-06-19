from inference_engine.api.protocol import (
    ChatCompletionRequest, ChatCompletionResponse, Choice, ResponseMessage, Usage,
)


def test_request_parses_and_defaults():
    req = ChatCompletionRequest.model_validate(
        {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert req.model == "qwen"
    assert req.messages[0].role == "user"
    assert req.max_tokens is None
    assert req.temperature is None


def test_response_shape():
    resp = ChatCompletionResponse(
        id="chatcmpl-x", created=123, model="qwen",
        choices=[Choice(message=ResponseMessage(content="hello"), finish_reason="stop")],
        usage=Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )
    dumped = resp.model_dump()
    assert dumped["object"] == "chat.completion"
    assert dumped["choices"][0]["message"]["role"] == "assistant"
    assert dumped["choices"][0]["finish_reason"] == "stop"
    assert dumped["usage"]["total_tokens"] == 4
