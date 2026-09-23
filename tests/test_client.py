from __future__ import annotations

from typing import Any

import pytest

from review_llm_eval import client as client_module
from review_llm_eval.client import DEFAULT_OPTIONS, OllamaClient, parse_chat_response


def test_parse_chat_response_converts_nanoseconds() -> None:
    body = {
        "message": {"role": "assistant", "content": "{}"},
        "done_reason": "stop",
        "total_duration": 2_500_000_000,
        "load_duration": 1_000_000,
        "prompt_eval_count": 480,
        "eval_count": 120,
        "eval_duration": 2_000_000_000,
    }
    resp = parse_chat_response(body)
    assert resp.content == "{}"
    assert resp.total_duration_s == pytest.approx(2.5)
    assert resp.eval_duration_s == pytest.approx(2.0)
    assert resp.eval_count == 120 and resp.done_reason == "stop"


def test_parse_chat_response_without_content_raises() -> None:
    with pytest.raises(ValueError):
        parse_chat_response({"error": "model not found"})


def test_chat_sends_schema_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"message": {"content": '{"ok": true}'}, "done_reason": "stop"}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    schema = {"type": "object"}
    resp = OllamaClient().chat("gemma3:4b", [{"role": "user", "content": "hola"}], schema, {})
    assert resp.content == '{"ok": true}'
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["format"] == schema
    assert captured["payload"]["stream"] is False


def test_default_options_are_deterministic_settings() -> None:
    assert DEFAULT_OPTIONS["temperature"] == 0
    assert isinstance(DEFAULT_OPTIONS["seed"], int)


def test_tests_cannot_reach_a_real_server() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        OllamaClient().version()
