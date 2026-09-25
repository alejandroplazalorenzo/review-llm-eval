from __future__ import annotations

from pathlib import Path

import pytest

from review_llm_eval.evaluate import evaluate_outputs, load_gold, main, pending_message


def gold_row(opinions: list[tuple[str, str]], staff: list[str], alerts: list[str], **kw) -> dict:
    return {
        "opinions": [{"topic": t, "polarity": p} for t, p in opinions],
        "staff": staff,
        "alerts": alerts,
        "incident": kw.get("incident"),
        "recommends": kw.get("recommends", False),
        "sentiment": "NEG",
    }


def output(opinions: list[tuple[str, str]], staff: list[str], **flags) -> dict:
    return {
        "opinions": [{"topic": t, "polarity": p, "quote": None} for t, p in opinions],
        "staff": staff,
        "incident": flags.pop("incident", None),
        "recommends": flags.pop("recommends", False),
        **{
            a: flags.get(a, False)
            for a in ("says_no_return", "legal_action", "theft", "illness", "bad_faith")
        },
    }


def test_evaluate_outputs_known_answers() -> None:
    gold = {
        1: gold_row([("room", "negative"), ("food", "negative")], ["Marta"], ["says_no_return"]),
        2: gold_row([("staff_service", "positive")], [], [], recommends=True),
        3: gold_row([("beach", "negative")], [], [], incident="long_wait"),
    }
    outputs = {
        # room pair right, food missed, noise invented; Marta right; alert right
        1: output([("room", "negative"), ("noise", "negative")], ["marta"], says_no_return=True),
        # right topic, wrong polarity; recommends missed; a false theft alert
        2: output([("staff_service", "negative")], ["Pedro"], theft=True),
        3: None,  # the model failed: counted as an empty answer
    }
    ev = evaluate_outputs(gold, outputs)
    assert ev.n == 3
    # topics: TP room, staff_service; FP noise; FN food, beach
    assert ev.topics.precision == pytest.approx(2 / 3) and ev.topics.recall == pytest.approx(2 / 4)
    # pairs: TP room-negative only
    assert ev.pairs.precision == pytest.approx(1 / 3) and ev.pairs.recall == pytest.approx(1 / 4)
    assert ev.staff.precision == 0.5 and ev.staff.recall == 1.0
    assert ev.alerts["says_no_return"].precision == 1.0
    assert ev.alerts["theft"].precision == 0.0
    assert ev.incident_accuracy == pytest.approx(2 / 3)
    assert ev.recommends_accuracy == pytest.approx(2 / 3)


def test_pending_message() -> None:
    assert pending_message(0) == "Human-labelled evaluation: pending (0/100 reviews labelled)"


def test_main_without_gold_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    main(["--gold", str(tmp_path / "missing.jsonl"), "--results", str(tmp_path / "res")])
    assert "pending (0/100" in capsys.readouterr().out
    assert not (tmp_path / "res").exists()
    assert load_gold(tmp_path / "missing.jsonl") == {}
