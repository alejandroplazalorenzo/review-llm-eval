"""SQLite store, layer 1 and the version contract."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
from pathlib import Path

from helpers import make_output
from review_llm_eval import layer1, store
from review_llm_eval.contract import BASE
from review_llm_eval.data import Review
from review_llm_eval.layer1 import LAYER1_VERSION, Layer1Result
from review_llm_eval.postprocess import expand

FULL = f"{LAYER1_VERSION}+qwen3:4b/prompt-v1"


class FakeAnalyzers:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, texts: list[str]) -> list[Layer1Result]:
        self.calls += 1
        return [Layer1Result("NEG" if "no" in t else "POS", 0.9, "others") for t in texts]


def open_store(tmp_path: Path, reviews: list[Review]) -> sqlite3.Connection:
    conn = store.connect(tmp_path / "s.sqlite")
    store.upsert_reviews(conn, reviews)
    return conn


def test_ingest_is_idempotent(tmp_path: Path, reviews: list[Review]) -> None:
    conn = store.connect(tmp_path / "s.sqlite")
    assert store.upsert_reviews(conn, reviews)[:3] == (10, 0, 0)
    assert store.upsert_reviews(conn, reviews)[:3] == (0, 0, 10)
    changed = [replace(reviews[0], text="texto nuevo"), *reviews[1:]]
    assert store.upsert_reviews(conn, changed)[:3] == (0, 1, 9)


def test_month_precision_from_wrote(tmp_path: Path, reviews: list[Review]) -> None:
    conn = open_store(tmp_path, [*reviews[:1], replace(reviews[1], wrote="")])
    rows = conn.execute("SELECT month, month_precision FROM review ORDER BY review_id").fetchall()
    assert tuple(rows[0]) == ("2021-03", "month")
    assert tuple(rows[1]) == (None, None)


def test_layer1_then_llm_pending(tmp_path: Path, reviews: list[Review]) -> None:
    conn = open_store(tmp_path, reviews)
    assert store.pending_llm(conn, store.ALL, None, LAYER1_VERSION) == []  # needs layer 1
    analyzers = FakeAnalyzers()
    assert layer1.run_layer1(conn, analyzers, store.ALL, False, FULL) == 10
    assert len(store.pending_llm(conn, store.ALL, None, LAYER1_VERSION)) == 10
    assert layer1.run_layer1(conn, analyzers, store.ALL, False, FULL) == 0  # idempotent
    language = conn.execute("SELECT DISTINCT language FROM review_enrichment").fetchall()
    assert [tuple(r) for r in language] == [("es",)]


def test_layer1_loads_nothing_when_nothing_is_pending(tmp_path: Path) -> None:
    conn = store.connect(tmp_path / "s.sqlite")

    def fail() -> None:
        raise AssertionError("analyzers must not be loaded")

    assert layer1.run_layer1(conn, None, store.ALL, False, FULL, load_analyzers=fail) == 0


def _write_llm(conn: sqlite3.Connection, review_id: int) -> None:
    e = expand(make_output(), BASE)
    store.write_llm(conn, review_id, e, ["marta"], 1, FULL)


def test_layer1_rewrite_empties_every_llm_column(tmp_path: Path, reviews: list[Review]) -> None:
    """Decision test: when layer 1 rewrites a row (text changed or --reprocess), the old
    LLM output must not survive until the LLM layer reaches the row again."""
    conn = open_store(tmp_path, reviews)
    layer1.run_layer1(conn, FakeAnalyzers(), store.ALL, False, FULL)
    _write_llm(conn, reviews[0].review_id)
    conn.commit()
    store.upsert_reviews(conn, [replace(reviews[0], text="otro texto"), *reviews[1:]])
    assert layer1.run_layer1(conn, FakeAnalyzers(), store.ALL, False, FULL) == 1
    row = conn.execute(
        "SELECT * FROM review_enrichment WHERE review_id = ?", (reviews[0].review_id,)
    ).fetchone()
    assert all(row[c] is None for c in store.LLM_COLUMNS)
    assert row["model_version"] == LAYER1_VERSION


def test_llm_columns_list_matches_what_the_llm_writes() -> None:
    import inspect

    source = inspect.getsource(store.write_llm)
    written = set(re.findall(r"(\w+) = \?", source)) - {"model_version", "enriched_at"}
    written.discard("review_id")
    assert written == set(store.LLM_COLUMNS)


def test_reprocess_picks_rows_of_other_versions(tmp_path: Path, reviews: list[Review]) -> None:
    conn = open_store(tmp_path, reviews)
    layer1.run_layer1(conn, FakeAnalyzers(), store.ALL, False, FULL)
    _write_llm(conn, reviews[0].review_id)
    conn.commit()
    newer = FULL.replace("prompt-v1", "prompt-v2")
    # without --reprocess nothing moves; with it, the old full version is redone
    assert layer1.run_layer1(conn, FakeAnalyzers(), store.ALL, False, newer) == 0
    assert layer1.run_layer1(conn, FakeAnalyzers(), store.ALL, True, newer) == 1


def test_scope_by_ids_and_hotel(tmp_path: Path, reviews: list[Review]) -> None:
    conn = open_store(tmp_path, reviews)
    scope = store.Scope(review_ids=frozenset({101, 203, 509}), hotel="Synthetic Resort A")
    assert layer1.run_layer1(conn, FakeAnalyzers(), scope, False, FULL) == 2  # 101, 203
