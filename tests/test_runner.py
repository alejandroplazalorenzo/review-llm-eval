"""Experiment runner, analysis helpers and the experiment registry (all offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import FakeClient, make_output
from review_llm_eval import analysis, experiments
from review_llm_eval.contract import BASE, Variant
from review_llm_eval.cues import has_cue
from review_llm_eval.data import Review
from review_llm_eval.runner import RunSpec, load_meta, load_records, prepare_server, run


def test_run_writes_records_and_resumes(tmp_path: Path, reviews: list[Review]) -> None:
    spec = RunSpec("t1", "TEST", "qwen3:4b", "s200", workers=2)
    client = FakeClient([make_output()])
    session = run(spec, reviews[:4], tmp_path, client=client, log=lambda _: None)
    assert session is not None and session.n_done == 4
    assert len(client.calls) == 5  # warm-up + 4
    records = load_records(tmp_path, "t1")
    assert [r["review_id"] for r in records] == [r.review_id for r in reviews[:4]]
    assert records[0]["extraction"]["output"] == make_output()
    # resume: only the missing ones are processed
    session = run(spec, reviews[:6], tmp_path, client=client, log=lambda _: None)
    assert session is not None and session.n_done == 2
    assert len(load_meta(tmp_path, "t1")["sessions"]) == 2


class FakeServer:
    def __init__(self, loaded: list[str]) -> None:
        self._loaded = loaded
        self.unloaded: list[str] = []

    def loaded(self) -> list[dict[str, Any]]:
        return [{"name": n} for n in self._loaded]

    def unload(self, model: str) -> None:
        self.unloaded.append(model)


def test_prepare_server_unloads_own_models_and_reports_foreign_ones() -> None:
    server = FakeServer(["qwen3:8b", "llama3:70b", "qwen3:4b"])
    unloaded, foreign = prepare_server(server, "qwen3:4b")  # type: ignore[arg-type]
    assert unloaded == ["qwen3:8b"] and server.unloaded == ["qwen3:8b"]
    assert foreign == ["llama3:70b"]  # never touched, only reported


def _rec(review_id: int, output: dict[str, Any] | None, *attempts: dict[str, Any]) -> dict:
    return {"review_id": review_id, "extraction": {"output": output, "attempts": list(attempts)}}


def _att(error: str | None = None, reason: str = "stop", tokens: int = 100) -> dict[str, Any]:
    return {
        "error": error,
        "done_reason": reason,
        "eval_count": tokens,
        "eval_duration_s": 1.0,
        "prompt_eval_count": 900,
        "latency_s": 2.0,
        "raw": "",
    }


def test_run_summary() -> None:
    records = [
        _rec(1, make_output(), _att()),
        _rec(2, make_output(), _att("cut", "length", 1024), _att()),
        _rec(3, None, _att("cut", "length", 1024), _att("cut", "length", 1024)),
        {"review_id": 4, "extraction": None, "network_error": "down"},
    ]
    s = analysis.run_summary(records, {"wall_s": 60.0, "n_done": 4})
    assert (s["valid"], s["valid_first_attempt"], s["retried"], s["rescued_by_retry"]) == (
        2,
        1,
        2,
        1,
    )
    assert s["failed"] == 1 and s["network_errors"] == 1 and s["runaway_reviews"] == 2
    assert s["reviews_per_min"] == 4.0
    assert s["decode_tok_per_s"] == pytest.approx((100 + 1024 + 100 + 2048) / 5)


def test_agreement_per_field() -> None:
    a = [_rec(1, make_output(), _att()), _rec(2, make_output(), _att())]
    b = [_rec(1, make_output(), _att()), _rec(2, make_output(nov=False, rsm="Otro."), _att())]
    ag = analysis.agreement(a, b, BASE, BASE)
    assert ag["n_pairs"] == 2 and ag["json"] == 0.5
    assert ag["opinions"] == 1.0 and ag["says_no_return"] == 0.5 and ag["summary"] == 0.5


def test_agreement_across_key_styles() -> None:
    long = Variant(keys="long")
    raw_long = {
        "opinions": [
            {
                "topic": "room",
                "polarity": "negative",
                "quote": "el aire acondicionado no funcionaba",
            },
            {"topic": "staff_service", "polarity": "positive", "quote": "Marta fue muy amable"},
        ],
        "staff": ["Marta"],
        "incident": "complaint_ignored",
        "says_no_return": True,
        "language": "spanish",
        "summary": "El aire acondicionado falló y nadie lo arregló.",
        "nights": 7,
        "with_children": True,
    }
    ag = analysis.agreement(
        [_rec(1, make_output(), _att())], [_rec(1, raw_long, _att())], BASE, long
    )
    assert ag["opinions"] == 1.0 and ag["incident"] == 1.0 and ag["summary"] == 1.0
    assert ag["json"] != ag["json"]  # NaN: raw JSON cannot be compared across key styles


def test_key_presence() -> None:
    records = [_rec(1, {"inc": None}, _att()), _rec(2, {"inc": "x"}, _att()), _rec(3, {}, _att())]
    p = analysis.key_presence(records, "inc")
    assert p["key_present"] == pytest.approx(2 / 3) and p["non_null_when_present"] == 0.5


def test_summary_echo_detection() -> None:
    echo = make_output(rsm="Resumen de 15 palabras: todo bien.")
    clean = make_output(rsm="Esperó 45 minutos en la recepción y nadie le atendió.")
    records = [
        {**_rec(1, echo, {**_att(), "raw": json.dumps(echo)}), "rating": 5},
        {**_rec(2, clean, {**_att(), "raw": json.dumps(clean)}), "rating": 5},
        _rec(
            3, None, {**_att("cut", "length"), "raw": '{"rsm": "máximo 30 palabras máximo 30 pal'}
        ),
    ]
    s = analysis.summary_stats(records, BASE, lambda rating: 8)
    assert s["echo_in_valid_summaries"] == 1  # a number of minutes is not an echo
    assert s["echo_in_first_raw"] == 2
    assert s["runaway_first_attempt"] == 1
    assert s["over_limit"] == 1  # the second summary has 10 words


def test_support_histogram() -> None:
    hist = dict(
        analysis.support_histogram([("hola mundo", "Hola  Mundo"), ("n/a", "x"), ("zzzz", "a")])
    )
    assert hist["1.0 (literal)"] == 1 and hist["[0.0, 0.1)"] == 1


def test_registry_covers_every_experiment() -> None:
    assert {"E1", "E2", "E3", "E5", "E6", "E7", "E8", "E10", "NICE", "L11"} <= set(experiments.RUNS)
    ids = [s.run_id for group in experiments.RUNS.values() for s in group]
    assert len(ids) == len(set(ids))
    e7 = {s.run_id: s.base_url for s in experiments.RUNS["E7"]}
    assert e7 == {"e7_ip_4b": "http://127.0.0.1:11434", "e7_localhost_4b": "http://localhost:11434"}
    assert [s.think for s in experiments.RUNS["E6"][:3]] == [False, None, True]


def test_maxlength_limits_are_derived_from_e1(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        experiments.maxlength_spec(tmp_path)
    rows = [
        _rec(
            i,
            make_output(ops=[{"t": "hab", "p": "neg", "lit": "x" * (10 + i)}], rsm="y" * (50 + i)),
            _att(),
        )
        for i in range(21)
    ]
    path = tmp_path / "e1_4b.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    spec = experiments.maxlength_spec(tmp_path)
    assert spec.variant.max_lengths == (29, 69)  # 95th percentiles


def test_subsets(reviews: list[Review]) -> None:
    assert [r.rating for r in experiments.subset("n5", reviews)] == [1, 2, 3, 4, 5]
    assert len(experiments.subset("s200", reviews)) == len(reviews)
    with pytest.raises(ValueError):
        experiments.subset("nope", reviews)


def test_analyze_all_writes_pending_reports_without_data(
    tmp_path: Path, reviews: list[Review], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiments, "RAW_PARQUET", tmp_path / "missing.parquet")
    results = tmp_path / "results"
    experiments.analyze_e1(tmp_path, results, reviews)
    experiments.analyze_e3(tmp_path, results)
    experiments.analyze_e4(tmp_path, results)
    experiments.analyze_e9(tmp_path, results)
    text = (results / "e1_model_choice.md").read_text(encoding="utf-8")
    assert "not run (pending)" in text
    assert "Pending." in (results / "e9_keep_alive.md").read_text(encoding="utf-8")


def test_e4_induced_retries_only_failed_first_attempts(
    tmp_path: Path, reviews: list[Review], monkeypatch: pytest.MonkeyPatch
) -> None:
    # e1_4b answers of 100..190 tokens: the cap is the 75th percentile (167); the reviews
    # kept are those above it by at most 20 % (170, 180, 190)
    rows = [
        {**_rec(r.review_id, make_output(), {**_att(), "eval_count": 100 + 10 * i}), "rating": 1}
        for i, r in enumerate(reviews)
    ]
    (tmp_path / "e1_4b.jsonl").write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    cap, chosen = experiments.induced_setup(tmp_path, reviews)
    assert cap == 167
    assert [r.review_id for r in chosen] == [r.review_id for r in reviews[7:]]

    experiments.run_e4_induced(reviews, tmp_path, FakeClient(['{"ops": ['], done_reason="length"))
    out = [
        json.loads(line) for line in (tmp_path / "e4_induced.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(out) == 3 and not out[0]["first_valid"]
    assert [d["temperature"] for d in out[0]["draws"]] == [0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    assert all(d["same_as_first"] for d in out[0]["draws"])


def test_alert_by_rating_counts_raised_and_phraseless(reviews: list[Review]) -> None:
    by_id = {r.review_id: r for r in reviews}
    records = [_rec(r.review_id, make_output(nov=r.rating == 1), _att()) for r in reviews]
    by = analysis.alert_by_rating(records, BASE, by_id, "says_no_return")
    assert set(by) == {1, 2, 3, 4, 5}
    valid, raised, phraseless = by[1]
    assert valid == 2 and raised == 2
    assert phraseless == sum(
        not has_cue("says_no_return", r.full_text) for r in reviews if r.rating == 1
    )
    assert by[5][1] == 0
