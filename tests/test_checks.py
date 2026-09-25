"""E1 objective checks and the regex cues they rely on."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from helpers import make_output
from review_llm_eval.checks import CHECKS, compare, compute
from review_llm_eval.cues import has_any_alert_cue, has_cue
from review_llm_eval.data import Review


def record(
    review_id: int, output: dict[str, Any] | None, *errors: str | None, reason: str = "stop"
) -> dict[str, Any]:
    attempts = [
        {"error": e, "done_reason": reason if e else "stop", "raw": "", "latency_s": 1.0}
        for e in errors
    ]
    return {"review_id": review_id, "extraction": {"output": output, "attempts": attempts}}


def test_ten_checks_are_declared() -> None:
    assert len(CHECKS) == 10


def test_compute_counts_each_failure_mode(reviews: list[Review]) -> None:
    five = replace(reviews[8], text="Todo genial, Marta muy amable.", rating=5)
    by_id = {r.review_id: r for r in reviews} | {five.review_id: five}
    bad = make_output(
        ops=[
            {"t": "atn", "p": "pos", "lit": "Marta muy amable"},
            {"t": "atn", "p": "pos", "lit": "no se menciona"},
            {"t": "buf", "p": "neg", "lit": "comida horrible y fría siempre"},
        ],
        emp=["Marta", "Pedro", "el camarero"],
        nov=True,
        leg=True,
    )
    records = [
        record(five.review_id, bad, None),
        record(reviews[0].review_id, None, "cut", "cut", reason="length"),
    ]
    c = compute(records, by_id)
    assert c["valid_first_attempt"] == 0.5 and c["runaway"] == 0.5
    assert c["duplicate_pairs"] == 1.0
    assert c["name_not_in_text"] == 0.5  # Pedro
    assert c["role_as_name"] == 1
    assert c["filler_quote"] == 1 / 3
    assert c["unsupported_quote"] == 0.5  # the food quote is invented
    assert c["no_return_5star_no_cue"] == 1
    assert c["legal_no_cue"] == 1
    assert c["never_varies"] == 11  # a single valid output: none of the 11 fields varies


def test_compare_respects_direction() -> None:
    a = {c.name: 0.0 for c in CHECKS} | {"valid_first_attempt": 1.0}
    b = {c.name: 1.0 for c in CHECKS} | {"valid_first_attempt": 1.0}
    verdicts = compare(a, b)
    assert verdicts[0] == "tie" and set(verdicts[1:]) == {"a"}
    assert compare(a | {"runaway": float("nan")}, b)[1] == "n/a"


def test_cues() -> None:
    assert has_cue("says_no_return", "No volveremos nunca a este hotel")
    assert has_cue("says_no_return", "NUNCA MÁS")
    assert not has_cue("says_no_return", "Volveremos pronto")
    assert has_cue("legal_action", "Pusimos una denuncia en la policía")
    assert has_cue("theft", "nos robaron de la caja fuerte")
    assert has_cue("illness", "intoxicación alimentaria")
    assert has_cue("bad_faith", "es una estafa")
    assert has_cue("nights", "estuvimos 7 noches") and has_cue("nights", "una semana")
    assert has_any_alert_cue("Una estafa") and not has_any_alert_cue("Todo perfecto")
