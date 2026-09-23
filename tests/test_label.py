from __future__ import annotations

from pathlib import Path

import pytest

from review_llm_eval.data import Review
from review_llm_eval.jsonl import read_jsonl
from review_llm_eval.label import (
    label_loop,
    parse_aspects,
    parse_sentiment,
    parse_would_return,
)


def test_parse_sentiment() -> None:
    assert parse_sentiment(" P ") == "positive"
    assert parse_sentiment("u") == "neutral"
    with pytest.raises(ValueError):
        parse_sentiment("good")


def test_parse_aspects_numbers_and_names() -> None:
    assert parse_aspects("1+ 4- 5~") == {
        "staff": "positive",
        "food": "negative",
        "pool_beach": "mixed",
    }
    assert parse_aspects("noise=, value-") == {"noise": "neutral", "value": "negative"}
    assert parse_aspects("") == {}


@pytest.mark.parametrize("bad", ["1", "99+", "spa+", "1+ 1-", "+"])
def test_parse_aspects_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_aspects(bad)


def test_parse_would_return() -> None:
    assert parse_would_return("y") is True
    assert parse_would_return("n") is False
    assert parse_would_return("") is None
    with pytest.raises(ValueError):
        parse_would_return("maybe")


def scripted(answers: list[str]):
    it = iter(answers)
    return lambda _prompt: next(it)


def test_label_loop_saves_each_review_and_resumes(tmp_path: Path, reviews: list[Review]) -> None:
    gold = tmp_path / "gold.jsonl"
    printed: list[str] = []
    # review 1: invalid sentiment first (re-asked), then valid answers; review 2: quit
    answers = ["x", "n", "3- 2-", "n", "q"]
    added = label_loop(reviews[:3], gold, "AP", scripted(answers), printed.append)
    assert added == 1
    rows = read_jsonl(gold)
    assert rows[0]["review_id"] == reviews[0].review_id
    assert rows[0]["sentiment"] == "negative"
    assert rows[0]["aspects"] == {"cleanliness": "negative", "room": "negative"}
    assert rows[0]["would_return"] is False and rows[0]["labeller"] == "AP"
    assert any("type p, n, m or u" in line for line in printed)
    # the labeller must not see the rating or the hotel
    assert not any(reviews[0].hotel in line for line in printed)

    # second session continues with the next review
    added = label_loop(reviews[:3], gold, "AP", scripted(["p", "", "", "q"]), lambda *_: None)
    assert added == 1
    assert [r["review_id"] for r in read_jsonl(gold)] == [r.review_id for r in reviews[:2]]
