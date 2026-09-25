"""Read the alerts the pipeline raised, one by one, and record whether the text really
says it. This is how the explicit-only alerts are calibrated for precision: small
batches read by a person, reported as "at least N correct", never as a rate.

Usage::

    python -m review_llm_eval.alerts_review --reviewer <your name>   # read and answer
    python -m review_llm_eval.alerts_review --report                  # results/alert_calibration.md

Verdicts go to ``gold/alert_verdicts.jsonl``. A model must not fill that file.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import textwrap
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_llm_eval import store
from review_llm_eval.config import MAIN_MODEL, RESULTS_DIR, STORE_PATH
from review_llm_eval.cues import ALERT_FIELDS
from review_llm_eval.jsonl import append_jsonl, iter_jsonl
from review_llm_eval.report import markdown_table

VERDICTS_PATH = Path("gold") / "alert_verdicts.jsonl"


def flagged(conn: sqlite3.Connection, version: str) -> list[tuple[int, str, str, str]]:
    """(review_id, alert, title, text) for every alert raised with ``version``."""
    rows = conn.execute(
        f"""SELECT r.review_id, r.title, r.text, {", ".join(f"e.{a}" for a in ALERT_FIELDS)}
            FROM review r JOIN review_enrichment e USING (review_id)
            WHERE e.model_version = ? ORDER BY r.review_id""",
        (version,),
    ).fetchall()
    return [(r[0], a, r[1], r[2]) for r in rows for a in ALERT_FIELDS if r[a]]


def load_verdicts(path: Path) -> dict[tuple[int, str], bool]:
    if not path.exists():
        return {}
    return {(v["review_id"], v["alert"]): bool(v["correct"]) for v in iter_jsonl(path)}


def summarize(
    hits: Sequence[tuple[int, str, str, str]], verdicts: dict[tuple[int, str], bool]
) -> list[list[Any]]:
    rows = []
    for alert in ALERT_FIELDS:
        keys = [(rid, a) for rid, a, _, _ in hits if a == alert]
        read = [k for k in keys if k in verdicts]
        correct = sum(verdicts[k] for k in read)
        rows.append([alert, len(keys), len(read), f"at least {correct}" if read else "pending"])
    return rows


def review_loop(
    hits: Sequence[tuple[int, str, str, str]],
    path: Path,
    reviewer: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> int:
    done = load_verdicts(path)
    added = 0
    for rid, alert, title, text in hits:
        if (rid, alert) in done:
            continue
        print_fn("\n" + "=" * 78)
        print_fn(f"review {rid} - alert: {alert}")
        print_fn(textwrap.fill(f"TITLE: {title}", 78))
        print_fn(textwrap.fill(text, 78))
        while True:
            answer = input_fn("Does the text state it in words? [y/n, q quits]: ").strip().lower()
            if answer in ("y", "n", "q"):
                break
        if answer == "q":
            break
        append_jsonl(
            path,
            {
                "review_id": rid,
                "alert": alert,
                "correct": answer == "y",
                "reviewer": reviewer,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        added += 1
    return added


def main(argv: Sequence[str] | None = None) -> None:
    from review_llm_eval.enrich import full_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    parser.add_argument("--model", default=MAIN_MODEL)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    conn = store.connect(args.store)
    hits = flagged(conn, full_version(args.model))
    if args.reviewer:
        review_loop(hits, args.verdicts, args.reviewer)
    table = markdown_table(
        ["alert", "raised", "read by a person", "correct"],
        summarize(hits, load_verdicts(args.verdicts)),
    )
    print(table)
    if args.report:
        args.results.mkdir(parents=True, exist_ok=True)
        (args.results / "alert_calibration.md").write_text(
            "# Alert calibration (human reading)\n\n"
            f"Alerts raised by `{full_version(args.model)}` in the pipeline store, read one by "
            "one by a person. Reported as 'at least N correct', not as a rate.\n\n"
            f"{table}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
