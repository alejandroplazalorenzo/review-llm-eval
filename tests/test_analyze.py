from __future__ import annotations

import math
from dataclasses import replace

import pytest

from helpers import make_output, ok_row
from review_llm_eval.analyze import (
    agreement,
    disagreements,
    example_outputs,
    field_stats,
    has_return_cue,
    looks_spanish,
    proxy_label,
    proxy_scores,
    reliability,
)
from review_llm_eval.data import Review
from review_llm_eval.experiments import optional_presence, stability


@pytest.mark.parametrize(
    ("stars", "label"),
    [(1, "negative"), (2, "negative"), (3, None), (4, "positive"), (5, "positive")],
)
def test_proxy_label(stars: int, label: str | None) -> None:
    assert proxy_label(stars) == label


def test_proxy_scores_excludes_three_stars_and_failures(reviews: list[Review]) -> None:
    sample = {r.review_id: r for r in reviews}
    predicted = {
        101: "negative",
        102: "mixed",
        203: "negative",
        204: "negative",
        305: "positive",
        306: "mixed",
        407: "positive",
        408: "positive",
        509: "positive",
    }
    results = {i: ok_row(i, make_output(s)) for i, s in predicted.items()}
    results[510] = {"review_id": 510, "status": "failed", "output": None}
    scores = proxy_scores(results, sample)
    assert scores.n == 7  # 10 reviews - two 3-star - one failure
    assert scores.accuracy == pytest.approx(6 / 7)
    # rows: negative, positive; columns: positive, neutral, negative, mixed
    assert scores.confusion == [[0, 0, 3, 1], [3, 0, 0, 0]]
    assert scores.f1_positive == 1.0


def test_reliability_counts_retries_and_failures() -> None:
    first = ok_row(1, make_output())
    retried = ok_row(2, make_output(), n_attempts=2, latency_s=3.0)
    retried["attempts"] = [dict(first["attempts"][0], error="bad", done_reason="length")] * 2
    failed = ok_row(3, make_output(), status="failed", n_attempts=2, latency_s=5.0)
    failed["output"] = None
    runs = [{"wall_clock_s": 30.0, "n_reviews": 3}]
    rel = reliability([first, retried, failed], runs)
    assert (rel.n, rel.valid_first_attempt, rel.retried, rel.recovered_by_retry) == (3, 1, 2, 1)
    assert rel.failed == 1 and rel.validity_rate == pytest.approx(2 / 3)
    assert rel.hit_token_limit == 2
    assert rel.latency_mean_s == pytest.approx(3.0)
    assert rel.reviews_per_min == pytest.approx(6.0)


def test_agreement_is_kappa_per_item() -> None:
    a = {
        1: ok_row(1, make_output("positive", staff="positive")),
        2: ok_row(2, make_output("negative", food="negative")),
    }
    b = {
        1: ok_row(1, make_output("positive", staff="positive")),
        2: ok_row(2, make_output("negative", staff="negative")),
        3: ok_row(3, make_output("mixed")),  # only in b: ignored
    }
    items = {ag.item: ag for ag in agreement(a, b)}
    assert items["sentiment (4 classes)"].kappa == pytest.approx(1.0)
    assert items["sentiment (4 classes)"].n == 2
    assert items["aspect present: staff"].raw_agreement == 0.5
    assert (
        items["aspect present: staff"].rate_a == 0.5 and items["aspect present: staff"].rate_b == 1
    )
    assert math.isnan(items["aspect present: noise"].kappa)  # never mentioned by either
    assert items["aspect with opinion: staff"].raw_agreement == 0.5


def test_field_stats(reviews: list[Review]) -> None:
    sample = {r.review_id: r for r in reviews}
    # 101 says "No volveremos" (cue); 305 has no return cue at all
    results = {
        101: ok_row(101, make_output(would_return=False, complaint="x", staff="negative")),
        305: ok_row(305, make_output(would_return=True, food="positive", room="neutral")),
    }
    stats = field_stats(results, sample)
    assert stats["would_return_non_null"] == 1.0
    assert stats["reviews_without_return_cue"] == 1
    assert stats["would_return_non_null_without_cue"] == 1.0
    assert stats["complaint_non_null"] == 0.5
    assert stats["aspects_per_review"] == 1.5


@pytest.mark.parametrize(
    ("text", "cue"),
    [
        ("Volveremos seguro", True),
        ("Repetiría sin dudarlo", True),
        ("Nunca más en este hotel", True),
        ("Regresaremos el año que viene", True),
        ("La comida excelente y la playa preciosa", False),
    ],
)
def test_return_cue(reviews: list[Review], text: str, cue: bool) -> None:
    assert has_return_cue(replace(reviews[0], title="", text=text)) is cue


def test_disagreements_flags_proxy_misses_and_model_conflicts(reviews: list[Review]) -> None:
    sample = {r.review_id: r for r in reviews[:2]}  # two 1-star reviews
    results = {
        "a": {101: ok_row(101, make_output("negative")), 102: ok_row(102, make_output("mixed"))},
        "b": {101: ok_row(101, make_output("negative")), 102: ok_row(102, make_output("negative"))},
    }
    rows = disagreements(["a", "b"], results, sample)
    assert [r["review_id"] for r in rows] == [102]
    assert rows[0]["proxy_miss"] == ["a"] and rows[0]["models_disagree"]


def test_stability_and_optional_presence() -> None:
    base = {1: ok_row(1, make_output("positive", complaint="c")), 2: ok_row(2, make_output())}
    rerun = {
        1: ok_row(1, make_output("positive", complaint="c")),
        2: ok_row(2, make_output("mixed")),
    }
    stats = stability(base, rerun)
    assert stats["n"] == 2 and stats["sentiment_identical"] == 0.5

    trimmed = make_output()
    del trimmed["complaint"], trimmed["would_return"]
    optional = {1: ok_row(1, trimmed), 2: ok_row(2, make_output(would_return=True))}
    presence = optional_presence(optional, base)
    assert presence["complaint_key_present"] == 0.5
    assert presence["would_return_key_present"] == 0.5
    assert presence["required_variant_complaint_non_null"] == 0.5
    assert presence["most_common_key_order_share"] == 0.5


def test_example_outputs_never_include_review_text(reviews: list[Review]) -> None:
    sample = {r.review_id: r for r in reviews}
    results = {"a": {i: ok_row(i, make_output()) for i in sample}}
    rows = example_outputs(["a"], results, sample, per_rating=1)
    assert [r["stars"] for r in rows] == [1, 2, 3, 4, 5]
    dumped = str(rows)
    assert all(r.text not in dumped and r.hotel not in dumped for r in reviews)


def test_looks_spanish() -> None:
    assert looks_spanish("Música nocturna muy fuerte y la comida fría")
    assert not looks_spanish("The room was dirty and the staff were rude")
    assert not looks_spanish("Buffet")  # no evidence either way -> not flagged
