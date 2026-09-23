"""Minimal Ollama ``/api/chat`` client with structured outputs.

``ChatClient`` is the seam used by the tests: anything with a matching ``chat``
method can stand in for the real server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests

from review_llm_eval.config import OLLAMA_URL, SEED

NS_PER_S = 1e9

DEFAULT_OPTIONS: dict[str, Any] = {
    "temperature": 0,
    "seed": SEED,
    "num_ctx": 4096,
    "num_predict": 512,
}
"""Kept identical across requests and models: changing ``num_ctx`` between requests
makes Ollama reload the model."""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    done_reason: str | None = None
    total_duration_s: float | None = None
    load_duration_s: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_s: float | None = None


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
    ) -> ChatResponse: ...


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
    ) -> ChatResponse:
        payload = {
            "model": model,
            "messages": messages,
            "format": schema,
            "options": options,
            "stream": False,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return parse_chat_response(resp.json())

    def version(self) -> str:
        resp = requests.get(f"{self.base_url}/api/version", timeout=10)
        resp.raise_for_status()
        return str(resp.json().get("version", "unknown"))


def _seconds(value: Any) -> float | None:
    return None if value is None else float(value) / NS_PER_S


def parse_chat_response(body: dict[str, Any]) -> ChatResponse:
    """Map the JSON body of a non-streaming ``/api/chat`` response."""
    message = body.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("response has no message.content")
    return ChatResponse(
        content=content,
        done_reason=body.get("done_reason"),
        total_duration_s=_seconds(body.get("total_duration")),
        load_duration_s=_seconds(body.get("load_duration")),
        prompt_eval_count=body.get("prompt_eval_count"),
        eval_count=body.get("eval_count"),
        eval_duration_s=_seconds(body.get("eval_duration")),
    )
