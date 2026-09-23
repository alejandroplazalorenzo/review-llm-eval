"""Classify one review: call the model, validate, retry once, record everything."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from review_llm_eval.client import ChatClient, ChatResponse
from review_llm_eval.data import Review
from review_llm_eval.prompt import PROMPT_VERSION, build_messages, build_retry_messages
from review_llm_eval.schema import JsonObject, parse_and_validate

Status = Literal["ok", "failed"]

MAX_RAW_ECHO_CHARS = 2000
"""An invalid answer echoed back in the retry is capped (runaway output can be long)."""


@dataclass(slots=True)
class AttemptLog:
    latency_s: float
    error: str | None
    done_reason: str | None = None
    total_duration_s: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_s: float | None = None


@dataclass(slots=True)
class ClassificationResult:
    review_id: int
    model: str
    prompt_version: str
    status: Status
    output: JsonObject | None
    raw: str | None
    attempts: list[AttemptLog] = field(default_factory=list)
    finished_at: str = ""

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def latency_s(self) -> float:
        return sum(a.latency_s for a in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["n_attempts"] = self.n_attempts
        row["latency_s"] = round(self.latency_s, 4)
        return row


def _log_from_response(latency_s: float, resp: ChatResponse, error: str | None) -> AttemptLog:
    return AttemptLog(
        latency_s=latency_s,
        error=error,
        done_reason=resp.done_reason,
        total_duration_s=resp.total_duration_s,
        prompt_eval_count=resp.prompt_eval_count,
        eval_count=resp.eval_count,
        eval_duration_s=resp.eval_duration_s,
    )


def classify_review(
    client: ChatClient,
    model: str,
    review: Review,
    schema: JsonObject,
    options: dict[str, Any],
    max_retries: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> ClassificationResult:
    """Classify ``review``; at most ``1 + max_retries`` model calls.

    Transport errors (HTTP, timeouts) and invalid answers both count as a failed
    attempt. Only a schema-invalid answer is echoed back in the retry prompt.
    """
    base_messages = build_messages(review)
    messages = base_messages
    result = ClassificationResult(
        review_id=review.review_id,
        model=model,
        prompt_version=PROMPT_VERSION,
        status="failed",
        output=None,
        raw=None,
    )
    for _ in range(1 + max_retries):
        start = clock()
        try:
            resp = client.chat(model, messages, schema, options)
        except Exception as exc:  # any transport error is recorded, never raised
            result.attempts.append(AttemptLog(clock() - start, f"request error: {exc!r}"))
            messages = base_messages
            continue
        latency = clock() - start
        obj, error = parse_and_validate(resp.content, schema)
        result.attempts.append(_log_from_response(latency, resp, error))
        result.raw = resp.content
        if obj is not None:
            result.status = "ok"
            result.output = obj
            break
        assert error is not None
        messages = build_retry_messages(base_messages, resp.content[:MAX_RAW_ECHO_CHARS], error)
    result.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    return result
