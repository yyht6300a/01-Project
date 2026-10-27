"""FastAPI 应用：OpenAI 兼容 /v1/chat/completions + /health。"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import Config
from .engine import make_engine
from .logger import log_error, log_request, setup as setup_logging
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ClassifierInput,
    Usage,
)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def create_app(cfg: Config) -> FastAPI:
    setup_logging(cfg.log_dir)
    engine = make_engine(cfg)
    app = FastAPI(title="safety-gate model server")

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": cfg.engine}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        started = time.time()
        try:
            inp = ClassifierInput.model_validate_json(req.user_content())
        except Exception as exc:
            log_error(f"bad user content: {exc}")
            return JSONResponse(status_code=400, content={"error": {
                "message": f"bad request: {exc}", "type": "invalid_request_error", "code": 400,
            }})

        try:
            gate = engine.classify(inp)
        except Exception as exc:
            log_error(f"engine error: {exc}")
            return JSONResponse(status_code=500, content={"error": {
                "message": f"engine error: {exc}", "type": "server_error", "code": 500,
            }})

        tool_for_log = inp.tool_name or (inp.proposed_tool_call.function if inp.proposed_tool_call else "")
        log_request(tool_for_log, gate, (time.time() - started) * 1000)
        content = gate.to_content()
        prompt_tokens = _estimate_tokens(req.user_content())
        completion_tokens = _estimate_tokens(content)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model or "safety-gate",
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens),
        )

    return app

