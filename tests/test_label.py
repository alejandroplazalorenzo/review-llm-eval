from __future__ import annotations

from pathlib import Path

import pytest

from review_llm_eval.data import Review
from review_llm_eval.jsonl import read_jsonl
from review_llm_eval.label import (
    label_loop,
    parse_alerts,
    parse_incident,
    parse_opinions,
    parse_sentiment,
    parse_staff,
    parse_yes_no,
)


def test_parse_sentiment() -> None:
    assert parse_sentiment(" P ") == "POS"
    assert parse_sentiment("u") == "NEU"
    with pytest.raises(ValueError):
        parse_sentiment("mixed")


def test_parse_opinions() -> None:
    assert parse_opinions("hab- atn+ buf=") == [
        {"topic": "room", "polarity": "negative"},
        {"topic": "staff_service", "polarity": "positive"},
        {"topic": "food", "polarity": "neutral"},
    ]
    assert parse_opinions("buf+ buf-") == [
        {"topic": "food", "polarity": "positive"},
        {"topic": "food", "polarity": "negative"},
    ]
    assert parse_opinions("") == []


@pytest.mark.parametrize("bad", ["hab", "spa+", "hab+ hab+", "+"])
def test_parse_opinions_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_opinions(bad)


def test_parse_staff_incident_alerts_yes_no() -> None:
    assert parse_staff(" Marta ,  José  Luis,") == ["Marta", "José Luis"]
    assert parse_incident("") is None and parse_incident("wait") == "long_wait"
    with pytest.raises(ValueError):
        parse_incident("queue")
    assert parse_alerts("rob nov") == ["says_no_return", "theft"]
    with pytest.raises(ValueError):
        parse_alerts("fire")
    assert parse_yes_no("y") is True and parse_yes_no("") is False


def scripted(answers: list[str]):
    it = iter(answers)
    return lambda _prompt: next(it)


def test_label_loop_saves_each_review_and_resumes(tmp_path: Path, reviews: list[Review]) -> None:
    gold = tmp_path / "gold.jsonl"
    printed: list[str] = []
    # review 1: invalid sentiment first (re-asked), then valid answers; review 2: quit
    answers = ["x", "n", "hab- rcp-", "", "ignored", "nov", "n", "q"]
    added = label_loop(reviews[:3], gold, "AP", scripted(answers), printed.append)
    assert added == 1
    row = read_jsonl(gold)[0]
    assert row["review_id"] == reviews[0].review_id and row["sentiment"] == "NEG"
    assert row["opinions"] == [
        {"topic": "room", "polarity": "negative"},
        {"topic": "front_desk", "polarity": "negative"},
    ]
    assert row["staff"] == [] and row["incident"] == "complaint_ignored"
    assert row["alerts"] == ["says_no_return"] and row["recommends"] is False
    assert any("type p, n or u" in line for line in printed)
    # the labeller never sees the rating or the hotel
    assert not any(reviews[0].hotel in line for line in printed)

    added = label_loop(reviews[:3], gold, "AP", scripted(["p", "", "", "", "", "y", "q"]), print)
    assert added == 1 and len(read_jsonl(gold)) == 2
