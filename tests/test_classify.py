from __future__ import annotations

import itertools
import json

import requests

from helpers import FakeClient, make_output
from review_llm_eval.classify import classify_review
from review_llm_eval.client import DEFAULT_OPTIONS
from review_llm_eval.data import Review
from review_llm_eval.schema import build_schema

SCHEMA = build_schema()
VALID = json.dumps(make_output("negative", complaint="Damp room", room="negative"))
INVALID_ENUM = json.dumps({**make_output(), "sentiment": "terrible"})
TRUNCATED = VALID[:30]


def fake_clock():
    """Each call advances one second, so every attempt 'takes' exactly 1 s."""
    counter = itertools.count()
    return lambda: float(next(counter))


def run(client: FakeClient, review: Review):
    return classify_review(client, "m", review, SCHEMA, DEFAULT_OPTIONS, clock=fake_clock())


def test_valid_first_attempt(reviews: list[Review]) -> None:
    client = FakeClient([VALID])
    result = run(client, reviews[0])
    assert result.status == "ok"
    assert result.n_attempts == 1
    assert result.output == json.loads(VALID)
    assert result.latency_s == 1.0
    assert len(client.calls) == 1


def test_retry_recovers_and_echoes_error(reviews: list[Review]) -> None:
    client = FakeClient([INVALID_ENUM, VALID])
    result = run(client, reviews[0])
    assert result.status == "ok"
    assert result.n_attempts == 2
    assert result.attempts[0].error is not None and "terrible" in result.attempts[0].error
    assert result.attempts[1].error is None
    retry_messages = client.calls[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["system", "user", "assistant", "user"]
    assert retry_messages[2]["content"] == INVALID_ENUM
    assert "rejected by the validator" in retry_messages[3]["content"]


def test_two_invalid_answers_are_recorded_as_failure(reviews: list[Review]) -> None:
    client = FakeClient([TRUNCATED, INVALID_ENUM])
    result = run(client, reviews[0])
    assert result.status == "failed"
    assert result.output is None
    assert result.n_attempts == 2
    assert result.raw == INVALID_ENUM  # last raw answer kept for inspection
    assert result.attempts[0].error.startswith("invalid JSON")


def test_max_retries_zero_means_single_call(reviews: list[Review]) -> None:
    client = FakeClient([INVALID_ENUM, VALID])
    result = classify_review(client, "m", reviews[0], SCHEMA, DEFAULT_OPTIONS, max_retries=0)
    assert result.status == "failed" and len(client.calls) == 1


def test_transport_error_then_success(reviews: list[Review]) -> None:
    client = FakeClient([requests.ConnectionError("refused"), VALID])
    result = run(client, reviews[0])
    assert result.status == "ok"
    assert result.attempts[0].error.startswith("request error")
    # a transport error is retried with the original prompt (nothing to echo back)
    assert [m["role"] for m in client.calls[1]["messages"]] == ["system", "user"]


def test_transport_errors_never_raise(reviews: list[Review]) -> None:
    client = FakeClient([TimeoutError("slow"), TimeoutError("slow")])
    result = run(client, reviews[0])
    assert result.status == "failed" and result.raw is None


def test_result_row_is_serialisable(reviews: list[Review]) -> None:
    row = run(FakeClient([VALID]), reviews[3]).to_dict()
    assert json.loads(json.dumps(row))["review_id"] == reviews[3].review_id
    assert row["n_attempts"] == 1 and row["prompt_version"]
