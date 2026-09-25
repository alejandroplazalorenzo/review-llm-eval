"""One review -> one validated model output: call, re-validate, retry once.

Two attempts with the same prompt: temperature 0, then 0.4. At temperature 0 a retry
would mostly reproduce the first failure: a generation that loops until ``num_predict``
runs out and leaves the JSON unclosed (with qwen3:8b on the public data, an opinion list
repeating the same items). Experiment E4 measures how many the retry rescues at each
temperature.

A transport error is not an invalid answer: it propagates as
``requests.RequestException`` and the caller counts it apart (the pipeline aborts after
N consecutive ones).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from review_llm_eval.client import Generation, LLMClient
from review_llm_eval.config import TEMPERATURES
from review_llm_eval.contract import BASE, JsonObject, Variant, build_schema
from review_llm_eval.data import Review
from review_llm_eval.prompt import build_prompt


@dataclass(slots=True)
class Attempt:
    temperature: float
    latency_s: float
    raw: str
    error: str | None
    done_reason: str | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    total_duration_s: float | None = None
    load_duration_s: float | None = None
    prompt_eval_duration_s: float | None = None
    eval_duration_s: float | None = None
    thinking_chars: int | None = None

    @classmethod
    def from_generation(
        cls, temperature: float, latency_s: float, gen: Generation, error: str | None
    ) -> Attempt:
        return cls(
            temperature=temperature,
            latency_s=latency_s,
            raw=gen.response,
            error=error,
            done_reason=gen.done_reason,
            eval_count=gen.eval_count,
            prompt_eval_count=gen.prompt_eval_count,
            total_duration_s=gen.total_duration_s,
            load_duration_s=gen.load_duration_s,
            prompt_eval_duration_s=gen.prompt_eval_duration_s,
            eval_duration_s=gen.eval_duration_s,
            thinking_chars=None if gen.thinking is None else len(gen.thinking),
        )


@dataclass(slots=True)
class Extraction:
    output: JsonObject | None
    attempts: list[Attempt] = field(default_factory=list)
    warning: str | None = None

    @property
    def raw(self) -> str | None:
        """Raw text of the last attempt (the valid one when ``output`` is set)."""
        return self.attempts[-1].raw if self.attempts else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "warning": self.warning,
            "attempts": [asdict(a) for a in self.attempts],
        }


def validator_for(schema: JsonObject) -> Draft202012Validator:
    return Draft202012Validator(schema)


def parse_and_validate(
    raw: str, validator: Draft202012Validator
) -> tuple[JsonObject | None, str | None]:
    """``(obj, None)`` if ``raw`` is JSON valid against the schema, else ``(None, reason)``."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"not JSON: {exc.msg} at char {exc.pos}"
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        err = errors[0]
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        return None, f"schema: {location}: {err.message}"[:500]
    return obj, None


def extract(
    client: LLMClient,
    model: str,
    review: Review,
    variant: Variant = BASE,
    temperatures: Sequence[float] = TEMPERATURES,
    clock: Callable[[], float] = time.perf_counter,
) -> Extraction:
    """Call the model with up to ``len(temperatures)`` attempts; stop at the first valid
    answer. Returns ``output=None`` plus a warning when every attempt is invalid."""
    schema = build_schema(variant)
    validator = validator_for(schema)
    prompt = build_prompt(review, variant)
    result = Extraction(output=None)
    reason = None
    for temperature in temperatures:
        start = clock()
        gen = client.generate(model, prompt, schema, temperature)
        latency = clock() - start
        obj, reason = parse_and_validate(gen.response, validator)
        result.attempts.append(Attempt.from_generation(temperature, latency, gen, reason))
        if obj is not None:
            result.output = obj
            return result
    result.warning = f"invalid answer after {len(temperatures)} attempts: {reason}"
    return result
