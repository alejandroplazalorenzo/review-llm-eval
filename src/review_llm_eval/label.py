"""Human labelling CLI: one review at a time, answers appended to ``gold/gold.jsonl``.

Usage: python -m review_llm_eval.label --labeller <your name>

The labeller never sees the star rating, the hotel or any model output (blind
labelling). Progress is saved after every review; run the command again to continue.
Labels must come from a person: a model cannot tell how often a model is wrong.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from review_llm_eval.config import GOLD_PATH, GOLD_TARGET, SAMPLE_PATH, SEED
from review_llm_eval.contract import CONTRACT_VERSION, INCIDENTS, TOPICS
from review_llm_eval.data import Review, gold_queue
from review_llm_eval.jsonl import append_jsonl, iter_jsonl, read_jsonl

SENTIMENT_KEYS = {"p": "POS", "n": "NEG", "u": "NEU"}
POLARITY_SIGNS = {"+": "positive", "-": "negative", "=": "neutral"}
ALERT_CODES = {
    "nov": "says_no_return",
    "leg": "legal_action",
    "rob": "theft",
    "enf": "illness",
    "fra": "bad_faith",
}

T = TypeVar("T")

HELP = (
    "Opinions: topic code + sign, separated by spaces (+ positive, - negative, = neutral).\n"
    "Example: 'hab- atn+ buf='. Empty line = no opinion.\n"
    + "  ".join(f"{code}={long}" for code, (long, _) in TOPICS.items())
)


class Quit(Exception):
    """Raised when the labeller types 'q'."""


def parse_sentiment(text: str) -> str:
    key = text.strip().lower()
    if key not in SENTIMENT_KEYS:
        raise ValueError("type p, n or u")
    return SENTIMENT_KEYS[key]


def parse_opinions(text: str) -> list[dict[str, str]]:
    """``'hab- atn+'`` -> [{topic: room, polarity: negative}, {topic: staff_service, ...}]."""
    out: list[dict[str, str]] = []
    for token in text.replace(",", " ").split():
        sign, code = token[-1], token[:-1].strip().lower()
        if sign not in POLARITY_SIGNS or code not in TOPICS:
            raise ValueError(f"'{token}': expected a topic code followed by + - =")
        pair = {"topic": TOPICS[code][0], "polarity": POLARITY_SIGNS[sign]}
        if pair in out:
            raise ValueError(f"'{token}': given twice")
        out.append(pair)
    return out


def parse_staff(text: str) -> list[str]:
    return [" ".join(n.split()) for n in text.split(",") if n.strip()]


def parse_incident(text: str) -> str | None:
    code = text.strip().lower() or "none"
    if code not in INCIDENTS:
        raise ValueError(f"type one of: {', '.join(INCIDENTS)} (empty = none)")
    return INCIDENTS[code][0]


def parse_alerts(text: str) -> list[str]:
    codes = text.replace(",", " ").split()
    unknown = [c for c in codes if c not in ALERT_CODES]
    if unknown:
        raise ValueError(f"unknown alert code(s): {', '.join(unknown)}")
    return sorted({ALERT_CODES[c] for c in codes})


def parse_yes_no(text: str) -> bool:
    key = text.strip().lower()
    if key in ("y", "s"):
        return True
    if key in ("n", ""):
        return False
    raise ValueError("type y or n (empty = n)")


def gold_record(review_id: int, labeller: str, **labels: Any) -> dict[str, Any]:
    return {
        "review_id": review_id,
        **labels,
        "labeller": labeller,
        "labelled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "contract_version": CONTRACT_VERSION,
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
                "Overall [p]ositive [n]egative ne[u]tral (q quits): ",
                parse_sentiment,
                input_fn,
                print_fn,
            )
            print_fn(HELP)
            opinions = ask("Opinions: ", parse_opinions, input_fn, print_fn)
            staff = ask(
                "Staff names written in the review (comma separated): ",
                parse_staff,
                input_fn,
                print_fn,
            )
            incident = ask(
                f"Incident ({'/'.join(INCIDENTS)}, empty = none): ",
                parse_incident,
                input_fn,
                print_fn,
            )
            alerts = ask(
                "Alerts stated in words (nov leg rob enf fra, empty = none): ",
                parse_alerts,
                input_fn,
                print_fn,
            )
            recommends = ask("Recommends it in words? [y/n]: ", parse_yes_no, input_fn, print_fn)
        except Quit:
            break
        append_jsonl(
            gold_path,
            gold_record(
                review.review_id,
                labeller,
                sentiment=sentiment,
                opinions=opinions,
                staff=staff,
                incident=incident,
                alerts=alerts,
                recommends=recommends,
            ),
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
