"""Build the evaluation sample: de-duplicate, then stratify by star rating.

Usage: python -m review_llm_eval.sample [--per-rating 70] [--seed 42]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from review_llm_eval.config import PER_RATING, RAW_PARQUET, SAMPLE_PATH, SEED
from review_llm_eval.data import load_parquet_rows, rows_to_reviews, stratified_sample
from review_llm_eval.jsonl import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PARQUET)
    parser.add_argument("--out", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--per-rating", type=int, default=PER_RATING)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = load_parquet_rows(args.raw)
    reviews = rows_to_reviews(rows)
    sample = stratified_sample(reviews, args.per_rating, args.seed)
    write_jsonl(args.out, (r.to_dict() for r in sample))

    counts = Counter(r.rating for r in sample)
    truncated = sum(r.truncated_in_source for r in sample)
    print(f"raw rows: {len(rows)} | after de-dup and cleaning: {len(reviews)}")
    print(f"sample: {len(sample)} reviews {dict(sorted(counts.items()))}")
    print(f"truncated in source (>= 780 chars): {truncated}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
