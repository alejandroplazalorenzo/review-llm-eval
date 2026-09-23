from __future__ import annotations

from pathlib import Path

import pytest

from helpers import make_output, ok_row
from review_llm_eval.evaluate import evaluate_model, load_gold, main, pending_message


def gold_row(review_id: int, sentiment: str, aspects: dict[str, str], would_return=None) -> dict:
    return {
        "review_id": review_id,
        "sentiment": sentiment,
        "aspects": aspects,
        "would_return": would_return,
    }


def test_evaluate_model_known_answers() -> None:
    gold = {
        1: gold_row(1, "negative", {"staff": "negative", "food": "negative"}, False),
        2: gold_row(2, "positive", {"staff": "positive"}, True),
        3: gold_row(3, "mixed", {"room": "negative"}),
    }
    results = {
        # staff TP (sentiment right), food FN, room FP
        1: ok_row(
            1, make_output("negative", would_return=False, staff="negative", room="negative")
        ),
        # staff TP (sentiment wrong)
        2: ok_row(2, make_output("positive", would_return=None, staff="mixed")),
        # model failed on review 3 -> counts as an empty answer
        3: {"review_id": 3, "status": "failed", "output": None},
    }
    ev = evaluate_model(gold, results)
    assert ev.n == 3
    assert ev.sentiment_accuracy == pytest.approx(2 / 3)
    assert ev.aspects["staff"].precision == 1.0 and ev.aspects["staff"].recall == 1.0
    assert ev.aspects["food"].recall == 0.0 and ev.aspects["food"].support == 1
    assert ev.aspects["room"].precision == 0.0  # FP on review 1, FN on review 3
    # micro: TP=2 (staff x2), FP=1 (room on 1), FN=2 (food on 1, room on 3)
    assert ev.aspects_micro.precision == pytest.approx(2 / 3)
    assert ev.aspects_micro.recall == pytest.approx(2 / 4)
    assert ev.aspect_sentiment_accuracy == pytest.approx(1 / 2)
    assert ev.would_return_accuracy == pytest.approx(2 / 3)  # 1 ok, 2 wrong, 3 null==null


def test_pending_message() -> None:
    assert pending_message(0) == "Human-labelled evaluation: pending (0/200 reviews labelled)"


def test_main_without_gold_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    main(["--gold", str(tmp_path / "missing.jsonl"), "--results", str(tmp_path / "res")])
    assert "pending (0/200" in capsys.readouterr().out
    assert not (tmp_path / "res").exists()
    assert load_gold(tmp_path / "missing.jsonl") == {}
