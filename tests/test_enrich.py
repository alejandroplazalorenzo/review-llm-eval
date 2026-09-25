"""The pipeline with a fake LLM: batches, commits, pending rows, network errors, gates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import requests

from helpers import FakeClient, make_output
from review_llm_eval import enrich, gates, layer1, store
from review_llm_eval.data import Review
from test_store import FakeAnalyzers, open_store

MODEL = "qwen3:4b"


def ready_store(tmp_path: Path, reviews: list[Review]) -> sqlite3.Connection:
    conn = open_store(tmp_path, reviews)
    layer1.run_layer1(conn, FakeAnalyzers(), store.ALL, False, enrich.full_version(MODEL))
    return conn


def versions(conn: sqlite3.Connection) -> dict[int, str]:
    return dict(conn.execute("SELECT review_id, model_version FROM review_enrichment"))


def test_full_version_string() -> None:
    assert enrich.full_version(MODEL) == f"{layer1.LAYER1_VERSION}+qwen3:4b/prompt-v1"


def test_enrich_writes_every_pending_row(tmp_path: Path, reviews: list[Review]) -> None:
    conn = ready_store(tmp_path, reviews)
    client = FakeClient([make_output(emp=["Marta", "el camarero"])])
    stats = enrich.enrich_llm(conn, client, MODEL, store.ALL, workers=3, batch_size=4)
    assert (stats.done, stats.failed_invalid, stats.failed_network) == (10, 0, 0)
    assert set(versions(conn).values()) == {enrich.full_version(MODEL)}
    row = conn.execute("SELECT * FROM review_enrichment WHERE review_id = 101").fetchone()
    assert json.loads(row["raw_output"]) == make_output(emp=["Marta", "el camarero"])
    assert json.loads(row["staff"]) == []  # "Marta" is not in review 101; the role is vetoed
    assert row["incident"] == "complaint_ignored" and row["llm_attempts"] == 1
    # nothing left: a second run makes no call
    again = FakeClient([make_output()])
    assert enrich.enrich_llm(conn, again, MODEL, store.ALL).done == 0
    assert again.calls == []


def test_invalid_rows_stay_pending_with_a_warning(tmp_path: Path, reviews: list[Review]) -> None:
    conn = ready_store(tmp_path, reviews)
    stats = enrich.enrich_llm(conn, FakeClient(["{not json"]), MODEL, store.ALL)
    assert stats.done == 0 and stats.failed_invalid == 10 and not stats.aborted
    assert "after 2 attempts" in stats.warnings[0]
    assert set(versions(conn).values()) == {layer1.LAYER1_VERSION}  # still pending
    assert len(store.pending_llm(conn, store.ALL, None, layer1.LAYER1_VERSION)) == 10


def test_consecutive_network_errors_abort(tmp_path: Path, reviews: list[Review]) -> None:
    conn = ready_store(tmp_path, reviews)
    client = FakeClient([requests.ConnectionError("refused")])
    stats = enrich.enrich_llm(conn, client, MODEL, store.ALL, workers=1, batch_size=25)
    assert stats.aborted and stats.failed_network == 5 and stats.done == 0
    assert "stopped answering" in stats.warnings[-1]


def test_commit_per_batch_makes_the_run_resumable(tmp_path: Path, reviews: list[Review]) -> None:
    conn = ready_store(tmp_path, reviews)
    # 4 good answers (first batch), then the server goes away for good
    script: list[object] = [make_output()] * 4 + [requests.ConnectionError("gone")] * 6
    stats = enrich.enrich_llm(conn, FakeClient(script), MODEL, store.ALL, workers=1, batch_size=4)
    assert stats.done == 4 and stats.aborted
    other = sqlite3.connect(tmp_path / "s.sqlite")  # what another process sees: committed rows
    done = other.execute(
        "SELECT count(*) FROM review_enrichment WHERE model_version = ?",
        (enrich.full_version(MODEL),),
    ).fetchone()[0]
    assert done == 4
    rest = enrich.enrich_llm(conn, FakeClient([make_output()]), MODEL, store.ALL)
    assert rest.done == 6


def test_main_runs_layer1_skips_llm_when_ollama_is_down_and_writes_gates(
    tmp_path: Path, reviews: list[Review], monkeypatch: pytest.MonkeyPatch
) -> None:
    open_store(tmp_path, reviews).close()
    monkeypatch.setattr(layer1, "PysentimientoAnalyzers", FakeAnalyzers)
    monkeypatch.setattr(
        enrich, "healthcheck", lambda client, model: (False, "Ollama does not answer")
    )
    code = enrich.main(
        [
            "--store", str(tmp_path / "s.sqlite"),
            "--results", str(tmp_path / "res"),
            "--parquet", str(tmp_path / "missing.parquet"),
        ]
    )  # fmt: skip
    assert code == 0
    report = (tmp_path / "res" / "gates.md").read_text(encoding="utf-8")
    assert "every review in scope has layer 1 | pass" in report
    assert "no rows with version" in report


def test_gates_catch_broken_output(tmp_path: Path, reviews: list[Review]) -> None:
    conn = ready_store(tmp_path, reviews)
    enrich.enrich_llm(conn, FakeClient([make_output()]), MODEL, store.ALL)
    version = enrich.full_version(MODEL)
    report = gates.run_gates(conn, version)
    names = {g.name: g for g in report.gates}
    assert names["no duplicated opinions"].ok
    assert names["every staff name is in its review"].ok
    # the fake answers says_no_return=true everywhere: on 4-5 star reviews that is an
    # inferred alert, and no text supports it
    assert not names["alerts on 4-5 star reviews backed by a phrase"].ok
    # break the stored data the way a regression would
    conn.execute('UPDATE review_enrichment SET staff = \'["camarero", "nadie aqui"]\'')
    conn.execute(
        "UPDATE review_enrichment SET opinions = "
        '\'[{"topic":"room","polarity":"negative","quote":"x"},'
        '{"topic":"room","polarity":"negative","quote":"inventado del todo"}]\''
    )
    report = gates.run_gates(conn, version)
    failed = {g.name for g in report.gates if not g.ok}
    assert {
        "no duplicated opinions",
        "no staff name is a job title",
        "every staff name is in its review",
        "literal quotes >= 95%",
    } <= failed


def test_remap_reapplies_postprocessing_without_calling_the_model(
    tmp_path: Path, reviews: list[Review]
) -> None:
    conn = ready_store(tmp_path, reviews)
    enrich.enrich_llm(conn, FakeClient([make_output()]), MODEL, store.ALL)
    conn.execute("UPDATE review_enrichment SET staff = '[\"inventado\"]', incident = NULL")
    assert enrich.remap(conn, MODEL) == 10
    row = conn.execute("SELECT staff, incident FROM review_enrichment WHERE review_id = 101")
    assert tuple(row.fetchone()) == ("[]", "complaint_ignored")


def test_field_that_never_varies_fails_only_when_the_data_says_it_should(
    tmp_path: Path, reviews: list[Review]
) -> None:
    conn = ready_store(tmp_path, reviews)
    enrich.enrich_llm(conn, FakeClient([make_output()]), MODEL, store.ALL)
    names = {g.name: g for g in gates.run_gates(conn, enrich.full_version(MODEL)).gates}
    # every answer says with_children=true and 'kid' phrases are rare in the fixtures
    assert names["`with_children` takes more than one value"].ok
    # recommends is always false while three fixtures do not mention it... and none does
    assert "triggering phrase" in names["`recommends` takes more than one value"].detail
