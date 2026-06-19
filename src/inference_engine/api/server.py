import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .protocol import (
    ChatCompletionRequest, ChatCompletionResponse, Choice, ResponseMessage, Usage,
)


def create_app(engine, default_max_tokens: int = 512) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages must not be empty",
                                    "type": "invalid_request_error"}},
            )
        messages = [m.model_dump() for m in req.messages]
        max_tokens = req.max_tokens if req.max_tokens is not None else default_max_tokens
        result = engine.generate(messages, max_tokens)
        prompt_tokens = result["prompt_tokens"]
        completion_tokens = result["completion_tokens"]
        resp = ChatCompletionResponse(
            id="chatcmpl-" + uuid.uuid4().hex,
            created=int(time.time()),
            model=req.model,
            choices=[Choice(
                message=ResponseMessage(content=result["text"]),
                finish_reason=result["finish_reason"],
            )],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return resp.model_dump()

    return app
