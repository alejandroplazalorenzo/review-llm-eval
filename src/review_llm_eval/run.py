"""Run one model over the sample and store raw per-review results.

Usage:
    python -m review_llm_eval.run --model qwen2.5:7b-instruct
    python -m review_llm_eval.run --model gemma3:4b --concurrency 4

Writes ``data/outputs/<model>[__<tag>].jsonl`` (one line per review) and a
``.run.json`` file with wall-clock time and settings. Existing results are kept and
already-processed reviews are skipped unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_llm_eval.classify import ClassificationResult, classify_review
from review_llm_eval.client import DEFAULT_OPTIONS, ChatClient, OllamaClient
from review_llm_eval.config import OUTPUTS_DIR, SAMPLE_PATH, model_slug
from review_llm_eval.data import Review
from review_llm_eval.jsonl import append_jsonl, iter_jsonl, read_jsonl
from review_llm_eval.prompt import PROMPT_VERSION, build_messages
from review_llm_eval.schema import SCHEMA_VERSION, JsonObject, SchemaVariant, build_schema


def output_path(model: str, tag: str | None = None, root: Path = OUTPUTS_DIR) -> Path:
    name = model_slug(model) + (f"__{tag}" if tag else "")
    return root / f"{name}.jsonl"


def run_batch(
    client: ChatClient,
    model: str,
    reviews: Sequence[Review],
    schema: JsonObject,
    options: dict[str, Any],
    concurrency: int,
    on_result: Callable[[ClassificationResult], None],
) -> None:
    """Classify ``reviews`` with ``concurrency`` worker threads.

    ``on_result`` is always called from the calling thread, so it can write files
    without locking.
    """
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(classify_review, client, model, review, schema, options)
            for review in reviews
        ]
        for future in as_completed(futures):
            on_result(future.result())


def warm_up(client: ChatClient, model: str, review: Review, schema: JsonObject) -> float:
    """One untimed call so that model loading is not counted as latency."""
    start = time.perf_counter()
    client.chat(model, build_messages(review), schema, DEFAULT_OPTIONS)
    return time.perf_counter() - start


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="first N reviews only")
    parser.add_argument("--schema-variant", choices=["required", "optional"], default="required")
    parser.add_argument("--tag", default=None, help="suffix for the output file name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    reviews = [Review.from_dict(row) for row in read_jsonl(args.sample)]
    if args.limit is not None:
        reviews = reviews_for_limit(reviews, args.limit)
    variant: SchemaVariant = args.schema_variant
    schema = build_schema(variant)
    out = output_path(args.model, args.tag, args.out_dir)
    if args.overwrite and out.exists():
        out.unlink()
    done = {row["review_id"] for row in iter_jsonl(out)} if out.exists() else set()
    todo = [r for r in reviews if r.review_id not in done]
    print(f"{args.model}: {len(todo)} to do, {len(done)} already in {out}")
    if not todo:
        return

    client = OllamaClient()
    warm_s = warm_up(client, args.model, todo[0], schema)
    print(f"warm-up call: {warm_s:.1f} s (not counted)")

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    counter = {"n": 0, "failed": 0}

    def on_result(result: ClassificationResult) -> None:
        append_jsonl(out, result.to_dict())
        counter["n"] += 1
        counter["failed"] += result.status == "failed"
        if counter["n"] % 25 == 0 or counter["n"] == len(todo):
            elapsed = time.perf_counter() - t0
            print(
                f"  {counter['n']}/{len(todo)} done, {counter['failed']} failed, "
                f"{counter['n'] / elapsed * 60:.1f} reviews/min",
                flush=True,
            )

    run_batch(client, args.model, todo, schema, DEFAULT_OPTIONS, args.concurrency, on_result)
    wall_s = time.perf_counter() - t0

    run_info = {
        "model": args.model,
        "tag": args.tag,
        "n_reviews": len(todo),
        "wall_clock_s": round(wall_s, 2),
        "reviews_per_min": round(len(todo) / wall_s * 60, 2),
        "concurrency": args.concurrency,
        "options": DEFAULT_OPTIONS,
        "schema_variant": variant,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "ollama_version": client.version(),
        "warm_up_s": round(warm_s, 2),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
    run_file = out.with_suffix(".run.json")
    sessions = json.loads(run_file.read_text(encoding="utf-8")) if run_file.exists() else []
    sessions.append(run_info)
    run_file.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    print(json.dumps(run_info, indent=2))


def reviews_for_limit(reviews: list[Review], limit: int) -> list[Review]:
    """First ``limit`` reviews in a rating-balanced order (round-robin over ratings),
    so that small experimental runs still cover every star rating."""
    by_rating: dict[int, list[Review]] = {}
    for review in reviews:
        by_rating.setdefault(review.rating, []).append(review)
    queues = [by_rating[k] for k in sorted(by_rating)]
    ordered: list[Review] = []
    index = 0
    while len(ordered) < min(limit, len(reviews)):
        for queue in queues:
            if index < len(queue) and len(ordered) < limit:
                ordered.append(queue[index])
        index += 1
    return ordered


if __name__ == "__main__":
    main()
