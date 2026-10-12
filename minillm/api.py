import json
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import Engine


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=16000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = "minillm"
    max_tokens: int = Field(default=48, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0, le=5)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    use_cache: bool = True
    seed: int = 42
    stream: bool = False


def create_app(run, cfg, stage):
    app = FastAPI(title="MiniLLM local laboratory", version="1.0")
    engine = Engine(run, cfg, stage)
    lock = threading.Lock()
    counters = {"requests": 0, "completion_tokens": 0, "generation_seconds": 0.0}
    log_path = run / "api_requests.jsonl"

    @app.get("/health")
    def health():
        return {"status": "ok", "stage": stage, "device": str(engine.device)}

    @app.get("/metrics")
    def metrics():
        with lock:
            return dict(counters)

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        # This teaching API implements a single user turn, with explicit validation.
        if request.stream or len(request.messages) != 1 or request.messages[0].role != "user":
            raise HTTPException(400, "This laboratory endpoint supports one user message and stream=false.")
        with lock:
            try:
                result = engine.generate(request.messages[0].content, max_new_tokens=request.max_tokens,
                                         temperature=request.temperature, top_p=request.top_p, top_k=request.top_k,
                                         use_cache=request.use_cache, seed=request.seed)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            counters["requests"] += 1
            counters["completion_tokens"] += result["completion_tokens"]
            counters["generation_seconds"] += result["seconds"]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": time.time(), "stage": stage, "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "seconds": result["seconds"], "settings": result["settings"]}) + "\n")
        return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": stage,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]}, "finish_reason": "stop" if result["token_ids"][-1] == engine.tokenizer.eos_token_id else "length"}],
                "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"], "total_tokens": result["prompt_tokens"]+result["completion_tokens"]}}
    return app
