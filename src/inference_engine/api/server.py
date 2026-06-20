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


def main():
    import argparse

    import uvicorn

    from ..engine import LLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    engine = LLM(args.model, device=args.device)
    app = create_app(engine, default_max_tokens=args.max_tokens)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
