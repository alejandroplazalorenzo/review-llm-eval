"""The enrichment pipeline: layer 1, then the LLM layer, then the quality gates.

Usage::

    python -m review_llm_eval.enrich --sample data/sample.jsonl
    python -m review_llm_eval.enrich --hotel "Iberostar Dominicana" --limit 50
    python -m review_llm_eval.enrich --sample data/sample.jsonl --reprocess

Behaviour carried over from production:

- layer 1 always runs first; the LLM only sees rows whose layer 1 is current;
- the LLM calls run in ``WORKERS`` threads, ``BATCH_SIZE`` rows per batch, one commit
  per batch (kill it and run it again: it continues from the last commit);
- a row whose answer is still invalid after the retry is left pending with a warning
  (it is retried by the next run); transport errors are counted apart and the run stops
  after ``MAX_CONSECUTIVE_NETWORK_ERRORS`` in a row;
- if Ollama does not answer the health check, the run goes on with layer 1 only;
- every row carries ``<layer1>+<model>/<prompt>``; ``--reprocess`` redoes rows enriched
  with any other version;
- the quality gates run at the end of every run (``gates.py``) and are written to
  ``results/gates.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from review_llm_eval import gates, layer1, store
from review_llm_eval.client import LLMClient, OllamaClient, healthcheck
from review_llm_eval.config import (
    BATCH_SIZE,
    MAIN_MODEL,
    MAX_CONSECUTIVE_NETWORK_ERRORS,
    MAX_WARNINGS_SHOWN,
    RAW_PARQUET,
    RESULTS_DIR,
    STORE_PATH,
    WORKERS,
)
from review_llm_eval.contract import BASE
from review_llm_eval.data import Review
from review_llm_eval.jsonl import read_jsonl
from review_llm_eval.postprocess import expand, hotel_veto, normalize_name, veto_for
from review_llm_eval.prompt import PROMPT_VERSION
from review_llm_eval.worker import Outcome, process_one, run_in_batches

log = logging.getLogger("review_llm_eval.enrich")


def llm_version(model: str) -> str:
    return f"{model}/{PROMPT_VERSION}"


def full_version(model: str) -> str:
    """``model_version`` of a row with both layers current."""
    return f"{layer1.LAYER1_VERSION}+{llm_version(model)}"


@dataclass(slots=True)
class LLMRunStats:
    done: int = 0
    failed_invalid: int = 0
    failed_network: int = 0
    aborted: bool = False
    warnings: list[str] = field(default_factory=list)


def _review_from_row(row: sqlite3.Row) -> Review:
    return Review(
        review_id=row["review_id"],
        rating=row["rating"],
        title=row["title"],
        text=row["text"],
        hotel=row["hotel"],
        wrote=row["wrote"] or "",
        truncated_in_source=bool(row["truncated_in_source"]),
    )


def enrich_llm(
    conn: sqlite3.Connection,
    client: LLMClient,
    model: str,
    scope: store.Scope,
    limit: int | None = None,
    workers: int = WORKERS,
    batch_size: int = BATCH_SIZE,
) -> LLMRunStats:
    """LLM layer over the pending rows of ``scope`` (commit per batch)."""
    stats = LLMRunStats()
    rows = store.pending_llm(conn, scope, limit, layer1.LAYER1_VERSION)
    if not rows:
        log.info("LLM layer: nothing pending (0 calls)")
        return stats
    reviews = [_review_from_row(r) for r in rows]
    base_veto = hotel_veto(store.hotel_names(conn))
    version = full_version(model)
    consecutive = 0
    started = time.monotonic()
    log.info("LLM layer: %d pending, model %s, %d thread(s)", len(reviews), model, workers)

    def handle(outcome: Outcome) -> bool:
        nonlocal consecutive
        review = outcome.review
        if outcome.extraction is None:
            consecutive += 1
            stats.failed_network += 1
            stats.warnings.append(f"review {review.review_id}: {outcome.network_error}")
            if consecutive >= MAX_CONSECUTIVE_NETWORK_ERRORS:
                stats.aborted = True
                stats.warnings.append(
                    f"Ollama stopped answering ({consecutive} errors in a row); LLM layer "
                    f"aborted after {stats.done} rows, the rest stay pending"
                )
                return False
            return True
        consecutive = 0
        extraction = outcome.extraction
        if extraction.output is None:
            stats.failed_invalid += 1
            stats.warnings.append(f"review {review.review_id}: {extraction.warning}")
            return True  # left pending: the next run retries it
        expanded = expand(
            extraction.output, BASE, review.full_text, veto_for(review.hotel, base_veto)
        )
        staff = [normalize_name(n) for n in expanded["staff"]]
        store.write_llm(conn, review.review_id, expanded, staff, len(extraction.attempts), version)
        stats.done += 1
        return True

    def end_batch(handed_out: int) -> None:
        conn.commit()
        rate = stats.done / max(time.monotonic() - started, 1e-9) * 60
        log.info("LLM layer: %d/%d (%.1f reviews/min)", handed_out, len(reviews), rate)

    run_in_batches(
        reviews,
        lambda r: process_one(client, model, r),
        workers,
        batch_size,
        handle,
        end_batch,
    )
    return stats


def remap(conn: sqlite3.Connection, model: str, scope: store.Scope = store.ALL) -> int:
    """Re-apply the current post-processing to the stored raw outputs, without the GPU
    (e.g. after changing a quote threshold or the name veto). Returns rows rewritten."""
    version = full_version(model)
    where, params = scope.sql(conn)
    rows = conn.execute(
        f"""SELECT r.review_id, r.hotel, r.rating, r.title, r.text, r.wrote,
                   r.truncated_in_source, e.raw_output, e.llm_attempts
            FROM review r JOIN review_enrichment e USING (review_id)
            WHERE e.model_version = ? AND e.raw_output IS NOT NULL {where}""",
        [version, *params],
    ).fetchall()
    base_veto = hotel_veto(store.hotel_names(conn))
    for row in rows:
        review = _review_from_row(row)
        expanded = expand(
            json.loads(row["raw_output"]), BASE, review.full_text, veto_for(review.hotel, base_veto)
        )
        staff = [normalize_name(n) for n in expanded["staff"]]
        store.write_llm(conn, review.review_id, expanded, staff, row["llm_attempts"], version)
    conn.commit()
    return len(rows)


def load_scope(sample: Path | None, hotel: str | None) -> store.Scope:
    ids = None
    if sample is not None:
        ids = frozenset(int(row["review_id"]) for row in read_jsonl(sample))
    return store.Scope(review_ids=ids, hotel=hotel)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    parser.add_argument("--sample", type=Path, default=None, help="only these review ids")
    parser.add_argument("--hotel", default=None, help="only this hotel")
    parser.add_argument("--limit", type=int, default=None, help="at most N rows per layer")
    parser.add_argument("--model", default=MAIN_MODEL)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--reprocess", action="store_true", help="redo rows of older versions")
    parser.add_argument("--only-layer1", action="store_true", help="skip the LLM layer")
    parser.add_argument(
        "--remap", action="store_true", help="re-apply post-processing to stored raw outputs"
    )
    parser.add_argument("--parquet", type=Path, default=RAW_PARQUET, help="for the golden gate")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    conn = store.connect(args.store)
    scope = load_scope(args.sample, args.hotel)
    version = full_version(args.model)
    if args.remap:
        log.info(
            "Re-mapped %d row(s) from their raw output (no model call)",
            remap(conn, args.model, scope),
        )
    n1 = layer1.run_layer1(conn, None, scope, args.reprocess, version, args.limit)
    log.info("Layer 1: %d row(s) written", n1)

    warnings: list[str] = []
    stats = LLMRunStats()
    if not args.only_layer1:
        client = OllamaClient()
        ok, warning = healthcheck(client, args.model)
        if not ok:
            warnings.append(warning or "health check failed")
        else:
            stats = enrich_llm(conn, client, args.model, scope, args.limit, args.workers)
            warnings.extend(stats.warnings[:MAX_WARNINGS_SHOWN])
            if len(stats.warnings) > MAX_WARNINGS_SHOWN:
                warnings.append(
                    f"... and {len(stats.warnings) - MAX_WARNINGS_SHOWN} more row warning(s)"
                )
    for warning in warnings:
        log.warning(warning)
    log.info(
        "LLM layer: %d ok, %d invalid after retry (pending), %d network errors%s",
        stats.done,
        stats.failed_invalid,
        stats.failed_network,
        " (aborted)" if stats.aborted else "",
    )

    report = gates.run_gates(conn, version, scope, args.parquet if args.parquet.exists() else None)
    path = gates.write_report(report, args.results / "gates.md", version)
    log.info("Quality gates: %s -> %s", "all green" if report.ok else "FAILURES", path)
    failed = stats.failed_invalid + stats.failed_network
    return 0 if report.ok and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
