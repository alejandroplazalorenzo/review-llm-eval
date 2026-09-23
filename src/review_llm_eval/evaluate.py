"""Score model outputs against the human gold labels in ``gold/gold.jsonl``.

Usage: python -m review_llm_eval.evaluate [--models qwen2.5:7b-instruct gemma3:4b]

This is the only place where the word "accuracy" means agreement with a human.
If the gold file does not exist yet it says so and exits without writing anything.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_llm_eval.config import (
    ASPECTS,
    DEFAULT_MODELS,
    GOLD_PATH,
    GOLD_TARGET,
    OUTPUTS_DIR,
    OVERALL_SENTIMENTS,
    RESULTS_DIR,
)
from review_llm_eval.jsonl import iter_jsonl
from review_llm_eval.metrics import PRF, accuracy, binary_prf, macro_f1
from review_llm_eval.report import markdown_table, write_csv
from review_llm_eval.run import output_path

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class HumanEval:
    n: int
    sentiment_accuracy: float
    sentiment_macro_f1: float
    aspects: dict[str, PRF]  # presence of each aspect
    aspects_micro: PRF
    aspect_sentiment_accuracy: float  # on aspects both human and model marked present
    would_return_accuracy: float


def load_gold(path: Path) -> dict[int, Row]:
    if not path.exists():
        return {}
    return {row["review_id"]: row for row in iter_jsonl(path)}


def evaluate_model(gold: dict[int, Row], results: dict[int, Row]) -> HumanEval:
    """Compare one model with the gold labels.

    Reviews the model failed on count as wrong (an empty answer): a failure is a
    real cost in production, so it is not silently dropped.
    """
    ids = sorted(gold)
    empty: Row = {"sentiment": None, "aspects": dict.fromkeys(ASPECTS), "would_return": None}
    outputs = [
        results[i]["output"] if i in results and results[i]["status"] == "ok" else empty
        for i in ids
    ]
    golds = [gold[i] for i in ids]

    y_true = [g["sentiment"] for g in golds]
    y_pred = [o["sentiment"] for o in outputs]

    per_aspect: dict[str, PRF] = {}
    tp_all = fp_all = fn_all = 0
    sentiment_hits: list[bool] = []
    for aspect in ASPECTS:
        tp = fp = fn = 0
        for g, o in zip(golds, outputs, strict=True):
            in_gold = aspect in g["aspects"]
            in_model = o["aspects"].get(aspect) is not None
            tp += in_gold and in_model
            fp += in_model and not in_gold
            fn += in_gold and not in_model
            if in_gold and in_model:
                sentiment_hits.append(g["aspects"][aspect] == o["aspects"][aspect])
        per_aspect[aspect] = binary_prf(tp, fp, fn)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn

    return HumanEval(
        n=len(ids),
        sentiment_accuracy=accuracy(y_true, y_pred),
        sentiment_macro_f1=macro_f1(y_true, y_pred, OVERALL_SENTIMENTS),
        aspects=per_aspect,
        aspects_micro=binary_prf(tp_all, fp_all, fn_all),
        aspect_sentiment_accuracy=(
            sum(sentiment_hits) / len(sentiment_hits) if sentiment_hits else float("nan")
        ),
        would_return_accuracy=accuracy(
            [g["would_return"] for g in golds], [o["would_return"] for o in outputs]
        ),
    )


def pending_message(n_labelled: int, target: int = GOLD_TARGET) -> str:
    return f"Human-labelled evaluation: pending ({n_labelled}/{target} reviews labelled)"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--min-labels", type=int, default=GOLD_TARGET)
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if len(gold) < args.min_labels:
        print(pending_message(len(gold)))
        if not gold:
            return
        print("(scoring the partial gold set anyway; numbers are provisional)")

    summary_headers = [
        "model", "n", "sentiment_accuracy", "sentiment_macro_f1", "aspect_micro_precision",
        "aspect_micro_recall", "aspect_micro_f1", "aspect_sentiment_accuracy",
        "would_return_accuracy",
    ]  # fmt: skip
    summary_rows: list[list[Any]] = []
    aspect_rows: list[list[Any]] = []
    for model in args.models:
        path = output_path(model, root=args.outputs)
        results = {row["review_id"]: row for row in iter_jsonl(path)} if path.exists() else {}
        ev = evaluate_model(gold, results)
        micro = ev.aspects_micro
        summary_rows.append(
            [
                model,
                ev.n,
                ev.sentiment_accuracy,
                ev.sentiment_macro_f1,
                micro.precision,
                micro.recall,
                micro.f1,
                ev.aspect_sentiment_accuracy,
                ev.would_return_accuracy,
            ]
        )
        for aspect, prf in ev.aspects.items():
            aspect_rows.append([model, aspect, prf.support, prf.precision, prf.recall, prf.f1])

    aspect_headers = ["model", "aspect", "gold_support", "precision", "recall", "f1"]
    write_csv(args.results / "human_eval_summary.csv", summary_headers, summary_rows)
    write_csv(args.results / "human_eval_aspects.csv", aspect_headers, aspect_rows)
    print(markdown_table(summary_headers, summary_rows))
    print()
    print(markdown_table(aspect_headers, aspect_rows))


if __name__ == "__main__":
    main()
