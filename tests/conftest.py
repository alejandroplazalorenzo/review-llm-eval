"""Shared fixtures. Everything here is synthetic; no test touches the network."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from review_llm_eval.data import Review
from review_llm_eval.jsonl import read_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def reviews() -> list[Review]:
    """Ten synthetic Spanish reviews, two per star rating."""
    return [Review.from_dict(row) for row in read_jsonl(FIXTURES / "reviews.jsonl")]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a real connection (e.g. to Ollama)."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
