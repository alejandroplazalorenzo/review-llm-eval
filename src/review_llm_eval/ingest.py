"""Load the public parquet into the SQLite store (extract -> normalize -> load).

Usage: python -m review_llm_eval.ingest

Idempotent: a second run over the same file writes nothing (0 inserted, 0 updated).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from review_llm_eval import store
from review_llm_eval.config import RAW_PARQUET, STORE_PATH
from review_llm_eval.data import load_parquet_rows, rows_to_reviews


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PARQUET)
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    args = parser.parse_args(argv)

    rows = load_parquet_rows(args.raw)
    reviews, warnings = rows_to_reviews(rows)
    conn = store.connect(args.store)
    inserted, updated, unchanged, load_warnings = store.upsert_reviews(conn, reviews)
    for warning in (warnings + load_warnings)[:20]:
        print(f"warning: {warning}")
    print(f"source rows: {len(rows)} | reviews after cleaning: {len(reviews)}")
    print(f"inserted {inserted}, updated {updated}, unchanged {unchanged} -> {args.store}")


if __name__ == "__main__":
    main()
