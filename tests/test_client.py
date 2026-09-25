from __future__ import annotations

from typing import Any

import pytest

from review_llm_eval import client as client_module
from review_llm_eval.client import OllamaClient, healthcheck, parse_generation
from review_llm_eval.config import OLLAMA_URL


class FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.body


def capture_post(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResponse(body)

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    return captured


def test_request_mirrors_the_production_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision test: /api/generate, one prompt, format = schema, think false,
    keep_alive 30m, temperature per attempt, num_ctx 4096, num_predict 1024, no seed."""
    captured = capture_post(monkeypatch, {"response": "{}", "done_reason": "stop"})
    schema = {"type": "object"}
    OllamaClient().generate("qwen3:4b", "reseña + instrucciones", schema, 0.4)
    payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["timeout"] == 300
    assert payload == {
        "model": "qwen3:4b",
        "prompt": "reseña + instrucciones",
        "stream": False,
        "format": schema,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.4, "num_ctx": 4096, "num_predict": 1024},
    }
    assert "seed" not in payload["options"] and "system" not in payload


def test_default_url_is_ipv4_loopback() -> None:
    assert OLLAMA_URL == "http://127.0.0.1:11434"


def test_think_and_keep_alive_can_be_left_out() -> None:
    payload = OllamaClient(think=None, keep_alive=None).payload("m", "p", {}, 0)
    assert "think" not in payload and "keep_alive" not in payload
    assert OllamaClient(think=True).payload("m", "p", {}, 0)["think"] is True


def test_parse_generation_converts_nanoseconds() -> None:
    gen = parse_generation(
        {
            "response": "{}",
            "thinking": "hmm",
            "done_reason": "length",
            "total_duration": 2_500_000_000,
            "load_duration": 1_000_000,
            "prompt_eval_count": 900,
            "prompt_eval_duration": 500_000_000,
            "eval_count": 120,
            "eval_duration": 2_000_000_000,
        }
    )
    assert gen.response == "{}" and gen.thinking == "hmm" and gen.done_reason == "length"
    assert gen.total_duration_s == pytest.approx(2.5)
    assert gen.prompt_eval_duration_s == pytest.approx(0.5)
    assert gen.eval_count == 120


def test_parse_generation_without_response_raises() -> None:
    with pytest.raises(ValueError):
        parse_generation({"error": "model not found"})


def test_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient()
    monkeypatch.setattr(client, "tags", lambda: [{"name": "qwen3:4b"}])
    assert healthcheck(client, "qwen3:4b") == (True, None)
    ok, warning = healthcheck(client, "qwen3:8b")
    assert not ok and warning and "layer 1 only" in warning


def test_healthcheck_when_the_server_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient()

    def refuse() -> list[dict[str, Any]]:
        raise client_module.requests.ConnectionError("refused")

    monkeypatch.setattr(client, "tags", refuse)
    ok, warning = healthcheck(client, "qwen3:4b")
    assert not ok and warning and "does not answer" in warning


def test_tests_cannot_reach_a_real_server() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        OllamaClient().version()
