"""Ollama ``/api/generate`` client with the production settings.

One request per attempt::

    POST {base_url}/api/generate
    {"model": ..., "prompt": <single prompt, review first>, "stream": false,
     "format": <JSON Schema>, "think": false, "keep_alive": "30m",
     "options": {"temperature": 0 | 0.4, "num_ctx": 4096, "num_predict": 1024}}

No seed, no system prompt, no chat turns. Every request is a plain ``requests.post``
(no pooled session), like production: this is what makes the localhost/127.0.0.1
difference visible on every call (experiment E7).

``LLMClient`` is the seam used by the tests: anything with a matching ``generate``
method can stand in for the server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests

from review_llm_eval.config import KEEP_ALIVE, NUM_CTX, NUM_PREDICT, OLLAMA_URL, TIMEOUT_S

NS_PER_S = 1e9


@dataclass(frozen=True, slots=True)
class Generation:
    response: str
    thinking: str | None = None
    done_reason: str | None = None
    total_duration_s: float | None = None
    load_duration_s: float | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_s: float | None = None
    eval_count: int | None = None
    eval_duration_s: float | None = None


class LLMClient(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
    ) -> Generation: ...


class OllamaClient:
    """``think`` and ``keep_alive`` are fixed per client; ``None`` leaves the field out of
    the request (used by experiments E6 and E9 only)."""

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        timeout_s: float = TIMEOUT_S,
        think: bool | None = False,
        keep_alive: str | None = KEEP_ALIVE,
        num_ctx: int = NUM_CTX,
        num_predict: int = NUM_PREDICT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.think = think
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def payload(
        self, model: str, prompt: str, schema: dict[str, Any], temperature: float
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
        }
        if self.think is not None:
            body["think"] = self.think
        if self.keep_alive is not None:
            body["keep_alive"] = self.keep_alive
        body["options"] = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        return body

    def generate(
        self, model: str, prompt: str, schema: dict[str, Any], temperature: float
    ) -> Generation:
        """One call. Transport and HTTP errors propagate as ``requests.RequestException``
        (the caller counts them apart from invalid answers)."""
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=self.payload(model, prompt, schema, temperature),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return parse_generation(resp.json())

    # --- metadata (recorded with every run) --------------------------------------------

    def version(self) -> str:
        resp = requests.get(f"{self.base_url}/api/version", timeout=10)
        resp.raise_for_status()
        return str(resp.json().get("version", "unknown"))

    def tags(self) -> list[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return list(resp.json().get("models", []))

    def loaded(self) -> list[dict[str, Any]]:
        """Models currently in memory (``ollama ps``)."""
        resp = requests.get(f"{self.base_url}/api/ps", timeout=10)
        resp.raise_for_status()
        return list(resp.json().get("models", []))

    def unload(self, model: str) -> None:
        """Ask the server to drop ``model`` from memory now (``keep_alive: 0``)."""
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=60,
        )
        resp.raise_for_status()


def _seconds(value: Any) -> float | None:
    return None if value is None else float(value) / NS_PER_S


def parse_generation(body: dict[str, Any]) -> Generation:
    """Map the JSON body of a non-streaming ``/api/generate`` response."""
    response = body.get("response")
    if not isinstance(response, str):
        raise ValueError("response has no 'response' text")
    thinking = body.get("thinking")
    return Generation(
        response=response,
        thinking=thinking if isinstance(thinking, str) else None,
        done_reason=body.get("done_reason"),
        total_duration_s=_seconds(body.get("total_duration")),
        load_duration_s=_seconds(body.get("load_duration")),
        prompt_eval_count=body.get("prompt_eval_count"),
        prompt_eval_duration_s=_seconds(body.get("prompt_eval_duration")),
        eval_count=body.get("eval_count"),
        eval_duration_s=_seconds(body.get("eval_duration")),
    )


def healthcheck(client: OllamaClient, model: str) -> tuple[bool, str | None]:
    """``(ok, warning)``: the server answers and has ``model``. When it fails the
    pipeline goes on with layer 1 only, like production."""
    try:
        names = [m.get("name", "") for m in client.tags()]
    except (requests.RequestException, ValueError) as exc:
        return False, (
            f"Ollama does not answer at {client.base_url} ({exc}); LLM layer skipped, layer 1 only"
        )
    if model not in names:
        return False, (
            f"Ollama has no model '{model}' (available: {', '.join(names) or 'none'}); "
            "LLM layer skipped, layer 1 only"
        )
    return True, None
