"""Test helpers (synthetic data and a scripted fake LLM client)."""

from __future__ import annotations

import copy
from typing import Any

from review_llm_eval.client import ChatResponse
from review_llm_eval.config import ASPECTS


def make_output(
    sentiment: str = "positive",
    complaint: str | None = None,
    would_return: bool | None = None,
    **aspects: str | None,
) -> dict[str, Any]:
    """A schema-valid model output; aspects not given are null."""
    return {
        "aspects": {a: aspects.get(a) for a in ASPECTS},
        "complaint": complaint,
        "would_return": would_return,
        "sentiment": sentiment,
    }


def ok_row(review_id: int, output: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A result row as written by ``run.py`` for a valid first-attempt answer."""
    row: dict[str, Any] = {
        "review_id": review_id,
        "status": "ok",
        "output": output,
        "raw": str(output),
        "n_attempts": 1,
        "latency_s": 1.0,
        "attempts": [
            {
                "latency_s": 1.0,
                "error": None,
                "done_reason": "stop",
                "total_duration_s": 0.9,
                "prompt_eval_count": 500,
                "eval_count": 100,
            }
        ],
    }
    row.update(extra)
    return row


class FakeClient:
    """Stand-in for ``OllamaClient``: returns (or raises) the scripted items in order
    and records every call."""

    def __init__(self, script: list[str | Exception]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
    ) -> ChatResponse:
        self.calls.append({"model": model, "messages": copy.deepcopy(messages), "options": options})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResponse(content=item, done_reason="stop", total_duration_s=0.5, eval_count=10)
