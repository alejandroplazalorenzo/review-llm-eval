"""Write a few pseudonymised pipeline outputs to ``results/example_outputs.jsonl``.

Only what the pipeline stored (topics, short quotes, flags, summary), never the review
text. Staff names are replaced by placeholders everywhere (names list, quotes, summary),
and any other capitalised word inside a sentence is masked too, so no person's name
reaches the committed file.

Usage: python -m review_llm_eval.examples [--per-rating 2]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from review_llm_eval import store
from review_llm_eval.config import MAIN_MODEL, RESULTS_DIR, STORE_PATH
from review_llm_eval.contract import BOOLEAN_FIELDS
from review_llm_eval.jsonl import write_jsonl
from review_llm_eval.postprocess import fold

_CAPITALISED = re.compile(r"(?<=[^\s.!?¡¿\"«(])(\s+)([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)")


MAX_QUOTE_CHARS = 80
"""Committed quotes are cut to this length: enough to show the shape, not the review."""


def placeholders(names: Sequence[str]) -> dict[str, str]:
    """One placeholder per distinct name (case and accents ignored), in order of appearance."""
    mapping: dict[str, str] = {}
    for name in names:
        key = fold(name)
        if key and key not in mapping:
            mapping[key] = f"[staff {len(mapping) + 1}]"
    return mapping


def pseudonymise(text: str | None, names: Sequence[str], limit: int | None = None) -> str | None:
    if text is None:
        return None
    mapping = placeholders(names)
    for name in sorted({n for n in names if fold(n)}, key=len, reverse=True):
        text = re.sub(re.escape(name), mapping[fold(name)], text, flags=re.IGNORECASE)
    text = _CAPITALISED.sub(lambda m: f"{m.group(1)}[name]", text)
    if limit is not None and len(text) > limit:
        text = text[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    return text


def example_rows(conn: sqlite3.Connection, version: str, per_rating: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT r.review_id, r.rating, r.month, r.month_precision, e.*
           FROM review r JOIN review_enrichment e USING (review_id)
           WHERE e.model_version = ? ORDER BY r.rating, r.review_id""",
        (version,),
    ).fetchall()
    picked: list[dict[str, Any]] = []
    for rating in (1, 2, 3, 4, 5):
        for row in [r for r in rows if r["rating"] == rating][:per_rating]:
            raw = json.loads(row["raw_output"])
            names = [*(raw.get("emp") or []), *json.loads(row["staff"])]
            mapping = placeholders(names)
            picked.append(
                {
                    "review_id": row["review_id"],
                    "rating": row["rating"],
                    "month": row["month"],
                    "month_precision": row["month_precision"],
                    "layer1": {
                        "sentiment": row["sentiment"],
                        "emotion": row["emotion"],
                        "language": row["language"],
                    },
                    "opinions": [
                        {**o, "quote": pseudonymise(o["quote"], names, MAX_QUOTE_CHARS)}
                        for o in json.loads(row["opinions"])
                    ],
                    "staff": [mapping[fold(n)] for n in json.loads(row["staff"])],
                    "incident": row["incident"],
                    **{b: bool(row[b]) for b in BOOLEAN_FIELDS},
                    "nights": row["nights"],
                    "summary": pseudonymise(row["summary"], names),
                    "model_version": row["model_version"],
                }
            )
    return picked


def star_text_gap(conn: sqlite3.Connection, version: str) -> dict[str, list[int]]:
    """Reviews whose stars and text disagree, seen by two independent layers: layer 1
    reads the text as negative (positive) and the LLM found no positive (negative)
    opinion, yet the guest gave 4-5 (1-2) stars. Review ids only."""
    rows = conn.execute(
        """SELECT r.review_id, r.rating, e.sentiment, e.opinions
           FROM review r JOIN review_enrichment e USING (review_id)
           WHERE e.model_version = ? ORDER BY r.review_id""",
        (version,),
    ).fetchall()
    gap: dict[str, list[int]] = {"high_stars_negative_text": [], "low_stars_positive_text": []}
    for row in rows:
        polarities = {o["polarity"] for o in json.loads(row["opinions"] or "[]")}
        if row["rating"] >= 4 and row["sentiment"] == "NEG" and "positive" not in polarities:
            gap["high_stars_negative_text"].append(row["review_id"])
        if row["rating"] <= 2 and row["sentiment"] == "POS" and "negative" not in polarities:
            gap["low_stars_positive_text"].append(row["review_id"])
    return gap


def main(argv: Sequence[str] | None = None) -> None:
    from review_llm_eval.enrich import full_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    parser.add_argument("--model", default=MAIN_MODEL)
    parser.add_argument("--per-rating", type=int, default=2)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    conn = store.connect(args.store)
    version = full_version(args.model)
    out = args.results / "example_outputs.jsonl"
    print(f"{write_jsonl(out, example_rows(conn, version, args.per_rating))} examples -> {out}")
    gap = star_text_gap(conn, version)
    n = conn.execute(
        "SELECT count(*) FROM review_enrichment WHERE model_version = ?", (version,)
    ).fetchone()[0]
    lines = [
        "# Stars vs text",
        "",
        f"Pipeline rows with version `{version}`: {n}. A review is listed when layer 1",
        "(pysentimiento) and the LLM opinions both contradict the star rating.",
        "",
        "| case | reviews | review ids (row index in the public parquet) |",
        "| --- | --- | --- |",
        *(
            f"| {k.replace('_', ' ')} | {len(v)} | {', '.join(map(str, v))} |"
            for k, v in gap.items()
        ),
        "",
    ]
    (args.results / "star_text_gap.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {args.results / 'star_text_gap.md'}")


if __name__ == "__main__":
    main()
