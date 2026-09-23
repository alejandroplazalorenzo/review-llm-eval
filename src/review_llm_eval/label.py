"""Human labelling CLI: one review at a time, answers appended to ``gold/gold.jsonl``.

Usage: python -m review_llm_eval.label --labeller <your name>

The labeller never sees the star rating, the hotel or any model output (blind
labelling), so the gold labels cannot be anchored on the things they will judge.
Progress is saved after every review; run the command again to continue.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from review_llm_eval.config import ASPECTS, GOLD_PATH, GOLD_TARGET, SAMPLE_PATH, SEED
from review_llm_eval.data import Review, gold_queue
from review_llm_eval.jsonl import append_jsonl, iter_jsonl, read_jsonl
from review_llm_eval.schema import SCHEMA_VERSION

SENTIMENT_KEYS = {"p": "positive", "n": "negative", "m": "mixed", "u": "neutral"}
ASPECT_SIGNS = {"+": "positive", "-": "negative", "~": "mixed", "=": "neutral"}
ASPECT_BY_NUMBER = {str(i): aspect for i, aspect in enumerate(ASPECTS, start=1)}

T = TypeVar("T")

HELP = (
    "Aspects: type number+sign, separated by spaces. Signs: + positive, - negative, "
    "~ mixed, = neutral. Example: '1+ 4- 5~'. Empty line = no aspect mentioned.\n"
    + "  ".join(f"{n}={a}" for n, a in ASPECT_BY_NUMBER.items())
)


class Quit(Exception):
    """Raised when the labeller types 'q'."""


def parse_sentiment(text: str) -> str:
    key = text.strip().lower()
    if key not in SENTIMENT_KEYS:
        raise ValueError("type p, n, m or u")
    return SENTIMENT_KEYS[key]


def parse_aspects(text: str) -> dict[str, str]:
    """Parse ``'1+ 4- 5~'`` (or ``'staff+ food-'``) into {aspect: sentiment}."""
    labels: dict[str, str] = {}
    for token in text.replace(",", " ").split():
        sign = token[-1]
        name = token[:-1].strip().lower()
        if sign not in ASPECT_SIGNS or not name:
            raise ValueError(f"'{token}': expected number or name followed by + - ~ =")
        aspect = ASPECT_BY_NUMBER.get(name, name)
        if aspect not in ASPECTS:
            raise ValueError(f"'{token}': unknown aspect")
        if aspect in labels:
            raise ValueError(f"'{token}': aspect given twice")
        labels[aspect] = ASPECT_SIGNS[sign]
    return labels


def parse_would_return(text: str) -> bool | None:
    key = text.strip().lower()
    if key in ("", "-"):
        return None
    if key in ("y", "s"):
        return True
    if key == "n":
        return False
    raise ValueError("type y, n or leave empty if not stated")


def gold_record(
    review_id: int,
    sentiment: str,
    aspects: dict[str, str],
    would_return: bool | None,
    labeller: str,
) -> dict[str, Any]:
    return {
        "review_id": review_id,
        "sentiment": sentiment,
        "aspects": {a: aspects[a] for a in ASPECTS if a in aspects},
        "would_return": would_return,
        "labeller": labeller,
        "labelled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
    }


def ask(
    prompt: str,
    parse: Callable[[str], T],
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
) -> T:
    """Ask until ``parse`` accepts the answer; 'q' quits."""
    while True:
        answer = input_fn(prompt)
        if answer.strip().lower() == "q":
            raise Quit
        try:
            return parse(answer)
        except ValueError as exc:
            print_fn(f"  ! {exc}")


def label_loop(
    queue: Sequence[Review],
    gold_path: Path,
    labeller: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> int:
    """Label the reviews of ``queue`` not yet in ``gold_path``; returns how many were added."""
    done = {row["review_id"] for row in iter_jsonl(gold_path)} if gold_path.exists() else set()
    added = 0
    for review in queue:
        if review.review_id in done:
            continue
        print_fn("\n" + "=" * 78)
        print_fn(f"[{len(done) + added + 1}/{len(queue)}] review {review.review_id}")
        print_fn(textwrap.fill(f"TITLE: {review.title}", 78))
        print_fn(textwrap.fill(review.text, 78))
        print_fn("-" * 78)
        try:
            sentiment = ask(
                "Overall [p]ositive [n]egative [m]ixed ne[u]tral (q quits): ",
                parse_sentiment,
                input_fn,
                print_fn,
            )
            print_fn(HELP)
            aspects = ask("Aspects: ", parse_aspects, input_fn, print_fn)
            would_return = ask(
                "Would return? [y]es [n]o, empty = not stated: ",
                parse_would_return,
                input_fn,
                print_fn,
            )
        except Quit:
            break
        append_jsonl(
            gold_path, gold_record(review.review_id, sentiment, aspects, would_return, labeller)
        )
        added += 1
    return added


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeller", required=True, help="your name or initials")
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--n", type=int, default=GOLD_TARGET)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sample = [Review.from_dict(row) for row in read_jsonl(args.sample)]
    queue = gold_queue(sample, args.n, SEED)
    added = label_loop(queue, args.gold, args.labeller)
    total = sum(1 for _ in iter_jsonl(args.gold)) if args.gold.exists() else 0
    print(f"\nlabelled this session: {added}; total in {args.gold}: {total}/{len(queue)}")


if __name__ == "__main__":
    main()
