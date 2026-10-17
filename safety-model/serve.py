"""OpenAI-compatible HTTP service for the tool-call risk model (the "PreTrained Model" box).

Frozen wire contract: ../接口/接口信息.md
  POST /v1/chat/completions  -- accepts and ignores response_format; reply content is the
      v2 4-field JSON: {"authorization", "risk_level", "risk_factors", "rationale"}.
  GET  /healthz              -- liveness + active backend (alias: /health)
  GET  /v1/models            -- OpenAI-style model list

Backends (same wire format, swap by flag):
  stub   rule-based placeholder (tool_metadata table + argument blacklist), no heavy
         deps, lets the gate integrate before training. NOT a safety claim: never put
         stub numbers into the report's results.
  model  Qwen2.5-1.5B-Instruct + optional LoRA adapter, greedy decode. Lazy-loads
         weights on first request unless --preload.

Usage (from safety-model/):
  py -3.10 serve.py                                  # stub on 127.0.0.1:8100
  py -3.10 serve.py --backend model --preload
  py -3.10 serve.py --backend model --adapter runs/qwen25-1.5b-risk-lora-v1/best

Every request appends one JSONL record (risk / latency_ms / tokens / prompt_version)
to runs/serve_requests.jsonl for the report's policy-cost column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_NAME = "Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # HF repo id (already in local cache)
PROMPT_VERSION = "v2"  # bump only if the frozen prompt/wire contract changes
DEFAULT_MAX_NEW_TOKENS = 384  # v2 answers run ~130-340 tokens

ARG_BLACKLIST_KEYS = ("password", "token", "otp", "credential", "secret")
ARG_BLACKLIST_VALUES = ("attacker",)  # placeholder marker; the real probe is the model


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _system_text(messages: list) -> str:
    for m in messages:
        if m.get("role") == "system":
            return m["content"]
    return ""


def _extract_tool_name(messages: list):
    for m in reversed(messages):
        if m.get("role") == "user":
            try:
                payload = json.loads(m["content"])
                if isinstance(payload, dict):
                    return payload.get("tool_name")
            except (ValueError, TypeError):
                pass
            break
    return None


class StubBackend:
    """Placeholder rule scorer per design doc section 6: metadata table + arg blacklist."""

    name = "stub"
    adapter = None

    def score(self, messages: list, max_new_tokens: int) -> dict:
        user_text = messages[-1]["content"] if messages else ""
        try:
            payload = json.loads(user_text)
        except (ValueError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            content = json.dumps({"authorization": "UNCERTAIN", "risk_level": "LOW", "risk_factors": [],
                                  "rationale": "stub: user payload is not valid JSON"})
            return {"content": content, "authorization": "UNCERTAIN", "risk_level": "LOW", "parse_ok": False,
                    "prompt_tokens": _approx_tokens(user_text), "completion_tokens": _approx_tokens(content),
                    "finish_reason": "stop"}

        meta = payload.get("tool_metadata") or {}
        read_only, remote = meta.get("read_only"), bool(meta.get("touches_remote"))
        if read_only is True:
            auth, level, why = "AUTHORIZED", "LOW", "read-only call, metadata says no side effects"
        elif read_only is False:
            auth, level = "UNCERTAIN", "LOW"
            why = "write call touching remote resources" if remote else "local write call"
        else:
            auth, level, why = "UNCERTAIN", "LOW", "no tool_metadata provided"

        haystack = json.dumps({k: v for k, v in payload.items() if k != "trace"}).lower()
        if any(k in haystack for k in ARG_BLACKLIST_KEYS) or any(v in haystack for v in ARG_BLACKLIST_VALUES):
            auth, level, why = "UNAUTHORIZED", "HIGH", "argument blacklist hit"

        content = json.dumps({"authorization": auth, "risk_level": level,
                              "risk_factors": [] if auth == "AUTHORIZED" else [why],
                              "rationale": f"stub (rule-based placeholder): {why}"})
        return {"content": content, "authorization": auth, "risk_level": level, "parse_ok": True,
                "prompt_tokens": _approx_tokens(user_text), "completion_tokens": _approx_tokens(content),
                "finish_reason": "stop"}


class ModelBackend:
    """Qwen2.5-1.5B-Instruct (+ optional LoRA adapter), greedy decode, frozen chat template."""

    name = "model"

    def __init__(self, base_model: str, adapter: str | None, device: str):
        self.base_model = base_model
        self.adapter = adapter
        self.device = device
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            from inference_utils import _flat_ids, load_model_and_tokenizer  # heavy imports, lazy
            import torch

            model, tokenizer, dev = load_model_and_tokenizer(self.base_model, self.adapter, self.device)
            model.eval()
            self._loaded = (model, tokenizer, dev, _flat_ids, torch)
        return self._loaded

    def score(self, messages: list, max_new_tokens: int) -> dict:
        import prompts

        model, tokenizer, dev, flat_ids, torch = self._load()
        prompt_ids = flat_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True))
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = gen[0, ids.shape[1]:]
        content = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        parsed = prompts.parse_answer(content)
        return {"content": content,
                "authorization": parsed["authorization"] if parsed else None,
                "risk_level": parsed["risk_level"] if parsed else None,
                "parse_ok": parsed is not None,
                "prompt_tokens": len(prompt_ids), "completion_tokens": int(new_ids.shape[0]),
                "finish_reason": "length" if int(new_ids.shape[0]) >= max_new_tokens else "stop"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "safety-model-serve/0.1"

    backend = None
    lock = threading.Lock()
    log_path = None
    started_at = 0.0

    def log_message(self, fmt, *args):  # silence default stderr chatter
        pass

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": message, "type": err_type, "code": None}})

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/healthz", "/health"):
            self._json(200, {
                "status": "ok",
                "backend": self.backend.name,
                "model": MODEL_NAME,
                "adapter": getattr(self.backend, "adapter", None),
                "prompt_version": PROMPT_VERSION,
                "uptime_s": round(time.time() - self.started_at, 1),
            })
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "ai6101-gp1"}]})
        else:
            self._error(404, f"not found: {self.path}", "not_found_error")

    def do_POST(self):
        try:
            self._do_post()
        except Exception as exc:  # never leak a broken socket to the gate
            try:
                self._error(500, f"unhandled server error: {exc}", "server_error")
            except Exception:
                pass

    def _do_post(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/v1/chat/completions":
            self._error(404, f"not found: {self.path}", "not_found_error")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8") if length > 0 else b"")
        except Exception as exc:
            self._error(400, f"invalid JSON body: {exc}")
            return
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            self._error(400, "field 'messages' must be a non-empty list")
            return
        messages = []
        for m in body["messages"]:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str):
                self._error(400, "every message needs a string 'content'")
                return
            messages.append({"role": str(m.get("role", "user")), "content": m["content"]})
        try:
            max_new = int(body.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS)
        except (TypeError, ValueError):
            max_new = DEFAULT_MAX_NEW_TOKENS
        # v2 answers run ~130-340 tokens; never truncate below the default.
        max_new = max(DEFAULT_MAX_NEW_TOKENS, min(max_new, 512))

        t0 = time.time()
        try:
            with self.lock:
                result = self.backend.score(messages, max_new)
        except Exception as exc:
            self._error(500, f"backend error: {exc}", "server_error")
            return
        latency_ms = round((time.time() - t0) * 1000, 1)

        payload = {
            "id": f"chatcmpl-safety-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or MODEL_NAME),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "finish_reason": result["finish_reason"],
            }],
            "usage": {
                "prompt_tokens": int(result["prompt_tokens"]),
                "completion_tokens": int(result["completion_tokens"]),
                "total_tokens": int(result["prompt_tokens"]) + int(result["completion_tokens"]),
            },
        }
        self._json(200, payload)

        tool_name = _extract_tool_name(messages)
        rationale = None
        try:
            rationale = json.loads(result["content"]).get("rationale")
        except (ValueError, TypeError):
            pass
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "id": payload["id"],
            "backend": self.backend.name,
            "tool_name": tool_name,
            "authorization": result["authorization"],
            "risk_level": result["risk_level"],
            "rationale": rationale,
            "parse_ok": result["parse_ok"],
            "latency_ms": latency_ms,
            "prompt_tokens": payload["usage"]["prompt_tokens"],
            "completion_tokens": payload["usage"]["completion_tokens"],
            "prompt_version": PROMPT_VERSION,
            "system_sha256": _sha12(_system_text(messages)),
        }
        print(f"[serve] {record['ts']} {self.backend.name} tool={tool_name} "
              f"auth={result['authorization']} level={result['risk_level']} "
              f"parse_ok={result['parse_ok']} {latency_ms}ms", flush=True)
        if self.log_path:
            try:
                with self.lock:
                    with open(self.log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"[serve] WARN: cannot append {self.log_path}: {exc}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible risk-model server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--backend", choices=["stub", "model"], default="stub")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (model backend)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preload", action="store_true", help="load weights before serving")
    parser.add_argument("--log-file", default="runs/serve_requests.jsonl")
    parser.add_argument("--no-log-file", action="store_true")
    args = parser.parse_args()

    if args.backend == "model":
        backend = ModelBackend(args.base_model, args.adapter, args.device)
        if args.preload:
            print("[serve] preloading model weights...", flush=True)
            backend._load()
    else:
        backend = StubBackend()

    Handler.backend = backend
    Handler.started_at = time.time()
    if args.no_log_file:
        Handler.log_path = None
    else:
        Handler.log_path = str(Path(args.log_file))
        Path(Handler.log_path).parent.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] backend={backend.name} model={MODEL_NAME} adapter={getattr(backend, 'adapter', None)}", flush=True)
    if backend.name == "stub":
        print("[serve] *** STUB backend: rule-based placeholder, not a safety model; "
              "do not report these numbers ***", flush=True)
    print(f"[serve] listening on http://{args.host}:{args.port} "
          f"(POST /v1/chat/completions, GET /healthz, GET /v1/models)", flush=True)
    print(f"[serve] request log -> {Handler.log_path or '(stdout only)'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[serve] shutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
