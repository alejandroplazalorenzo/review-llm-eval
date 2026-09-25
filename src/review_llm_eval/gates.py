"""Quality gates on the real output stored in SQLite. They run at the end of every
``enrich`` (and on demand with ``python -m review_llm_eval.gates``).

Each gate watches a failure that passes every schema check: a field stuck on one value,
duplicated opinions, quotes or names that are not in the review, alerts inferred from
tone. The thresholds are derived from the public data, as documented next to each
constant; none is a production figure.

Usage: python -m review_llm_eval.gates [--sample data/sample.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from review_llm_eval import store
from review_llm_eval.config import MAIN_MODEL, RAW_PARQUET, RESULTS_DIR, STORE_PATH
from review_llm_eval.contract import BOOLEAN_FIELDS
from review_llm_eval.cues import ALERT_FIELDS, CUES
from review_llm_eval.data import load_parquet_rows, rows_to_reviews
from review_llm_eval.postprocess import fold, is_literal, is_role_or_generic, name_in_text

LITERAL_QUOTES_MIN = 0.95
"""Share of kept quotes that must be literal. Derived from the E1 runs on the public data
(``results/quote_support.md``): literal quotes among the kept ones were 99.1 % for
qwen3:4b (673 of 679) and 98.3 % for qwen3:8b (834 of 848); the lower one, rounded down
to a multiple of 5 points, is 95 %. It catches a regression, not normal variation."""

PHRASELESS_ALERTS_PER_100 = 1
"""Alerts on 4-5 star reviews whose text contains no phrase for that alert: at most one per
hundred such reviews (always at least one, for a paraphrase the broad regex misses)."""

MIN_CUE_ROWS_FOR_VARIATION = 3
"""A field must take two values or more when at least this many rows in scope contain a
phrase that should trigger it."""

GOLDEN_TOLERANCE = 0.005
"""Mean rating per hotel, SQL on the store vs Python on the parquet: must agree to two
decimals."""


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class GateReport:
    gates: list[Gate] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(g.ok for g in self.gates)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.gates.append(Gate(name, bool(ok), detail))


def _rows(conn: sqlite3.Connection, version: str, scope: store.Scope) -> list[sqlite3.Row]:
    where, params = scope.sql(conn)
    return conn.execute(
        f"""SELECT r.review_id, r.rating, r.title, r.text, e.*
            FROM review r JOIN review_enrichment e USING (review_id)
            WHERE e.model_version = ? {where}""",
        [version, *params],
    ).fetchall()


def phraseless_alerts(rows: Sequence[sqlite3.Row]) -> int:
    """Alerts raised on a review whose text has no phrase for that alert (``cues.py``)."""
    return sum(
        bool(r[a]) and not CUES[a].search(fold(f"{r['title']}\n{r['text']}"))
        for r in rows
        for a in ALERT_FIELDS
    )


def run_gates(
    conn: sqlite3.Connection,
    version: str,
    scope: store.Scope = store.ALL,
    parquet: Path | None = None,
) -> GateReport:
    report = GateReport()
    _source_gates(conn, report, parquet)
    where, params = scope.sql(conn)
    no_layer1 = conn.execute(
        f"""SELECT count(*) FROM review r LEFT JOIN review_enrichment e USING (review_id)
            WHERE r.text <> '' AND e.sentiment IS NULL {where}""",
        params,
    ).fetchone()[0]
    report.add("every review in scope has layer 1", no_layer1 == 0, f"{no_layer1} missing")
    rows = _rows(conn, version, scope)
    if not rows:
        report.add("enriched rows", True, f"no rows with version {version} yet")
        return report
    n = len(rows)
    report.add("enriched rows", True, f"{n} rows with version {version}")

    # 1. Layer 1 of every enriched row matches the current text.
    stale = sum(r["content_hash"] != _current_hash(conn, r["review_id"]) for r in rows)
    report.add("LLM rows computed on the current text", stale == 0, f"{stale} stale")

    # 2. A field that never varies is broken (when the data says it should vary).
    texts = {r["review_id"]: f"{r['title']}\n{r['text']}" for r in rows}
    for name in ("incident", *BOOLEAN_FIELDS, "nights"):
        values = {r[name] if name != "nights" else r[name] is not None for r in rows}
        cue_rows = sum(CUES[name].search(fold(t)) is not None for t in texts.values())
        needed = cue_rows >= MIN_CUE_ROWS_FOR_VARIATION
        report.add(
            f"`{name}` takes more than one value",
            len(values) > 1 or not needed,
            f"{len(values)} value(s); {cue_rows} row(s) contain a triggering phrase"
            + ("" if needed else " (too few to require variation)"),
        )

    # 3. No duplicated (topic, polarity) pair survives the post-processing.
    opinions = {r["review_id"]: json.loads(r["opinions"] or "[]") for r in rows}
    dup = sum(
        len({(o["topic"], o["polarity"]) for o in ops}) != len(ops) for ops in opinions.values()
    )
    report.add("no duplicated opinions", dup == 0, f"{dup} row(s) with a repeated pair")

    # 4. Kept quotes are literal.
    kept = [(rid, o["quote"]) for rid, ops in opinions.items() for o in ops if o["quote"]]
    literal = sum(is_literal(q, texts[rid]) for rid, q in kept)
    share = literal / len(kept) if kept else 1.0
    report.add(
        f"literal quotes >= {LITERAL_QUOTES_MIN:.0%}",
        share >= LITERAL_QUOTES_MIN,
        f"{share:.1%} of {len(kept)} kept quotes",
    )

    # 5-6. Staff names: never a job title, always written in their own review.
    staff = {r["review_id"]: json.loads(r["staff"] or "[]") for r in rows}
    roles = sum(is_role_or_generic(name) for names in staff.values() for name in names)
    missing = sum(
        not name_in_text(name, texts[rid]) for rid, names in staff.items() for name in names
    )
    total_names = sum(len(v) for v in staff.values())
    report.add("no staff name is a job title", roles == 0, f"{roles} of {total_names}")
    report.add("every staff name is in its review", missing == 0, f"{missing} of {total_names}")

    # 7. Alerts on 4-5 star reviews need a phrase that supports them.
    positives = [r for r in rows if (r["rating"] or 0) >= 4]
    flagged = sum(any(r[a] for a in ALERT_FIELDS) for r in positives)
    phraseless = phraseless_alerts(positives)
    allowed = max(1, len(positives) * PHRASELESS_ALERTS_PER_100 // 100)
    report.add(
        "alerts on 4-5 star reviews backed by a phrase",
        phraseless <= allowed,
        f"{flagged} of {len(positives)} 4-5 star reviews flagged; {phraseless} alert(s) "
        f"without any supporting phrase (allowed {allowed})",
    )
    return report


def _current_hash(conn: sqlite3.Connection, review_id: int) -> str:
    row = conn.execute("SELECT content_hash FROM review WHERE review_id = ?", (review_id,))
    return str(row.fetchone()[0])


def _source_gates(conn: sqlite3.Connection, report: GateReport, parquet: Path | None) -> None:
    n_store = conn.execute("SELECT count(*) FROM review").fetchone()[0]
    no_month = conn.execute(
        "SELECT count(*) FROM review WHERE wrote IS NOT NULL AND month IS NULL"
    ).fetchone()[0]
    without_wrote = conn.execute("SELECT count(*) FROM review WHERE wrote IS NULL").fetchone()[0]
    report.add(
        "every dated review has month precision",
        no_month == 0,
        f"{no_month} unreadable; {without_wrote} of {n_store} have no date (month NULL)",
    )
    if parquet is None:
        report.add("golden: source counts and per-hotel mean", True, "skipped (no parquet)")
        return
    reviews, _ = rows_to_reviews(load_parquet_rows(parquet))
    report.add(
        "golden: rows in store == de-duplicated source rows",
        n_store == len(reviews),
        f"store {n_store}, source {len(reviews)}",
    )
    by_hotel: dict[str, list[int]] = defaultdict(list)
    for review in reviews:
        if review.rating is not None:
            by_hotel[review.hotel].append(review.rating)
    sql = dict(conn.execute("SELECT hotel, mean_rating FROM v_hotel").fetchall())
    diffs = [
        abs(sum(v) / len(v) - (sql.get(h) if sql.get(h) is not None else math.inf))
        for h, v in by_hotel.items()
    ]
    worst = max(diffs) if diffs else 0.0
    report.add(
        "golden: mean rating per hotel, SQL == Python",
        worst <= GOLDEN_TOLERANCE,
        f"{len(by_hotel)} hotels, largest difference {worst:.4f}",
    )


def write_report(report: GateReport, path: Path, version: str) -> Path:
    lines = [
        "# Quality gates",
        "",
        f"Version: `{version}`",
        "",
        "| Gate | Result | Detail |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {g.name} | {'pass' if g.ok else 'FAIL'} | {g.detail} |" for g in report.gates]
    lines += ["", f"**{'All gates pass' if report.ok else 'Some gates fail'}**", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    from review_llm_eval.enrich import full_version, load_scope

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--model", default=MAIN_MODEL)
    parser.add_argument("--parquet", type=Path, default=RAW_PARQUET)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    conn = store.connect(args.store)
    version = full_version(args.model)
    report = run_gates(
        conn,
        version,
        load_scope(args.sample, None),
        args.parquet if args.parquet.exists() else None,
    )
    path = write_report(report, args.results / "gates.md", version)
    for gate in report.gates:
        print(f"{'pass' if gate.ok else 'FAIL'}  {gate.name}: {gate.detail}")
    print(f"-> {path}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
