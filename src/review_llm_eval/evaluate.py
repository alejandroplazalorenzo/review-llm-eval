"""Score model outputs (and layer 1) against the human labels in ``gold/gold.jsonl``.

Usage: python -m review_llm_eval.evaluate [--runs e1_4b e1_8b]

This is the only place where precision, recall or accuracy mean agreement with a human.
If the gold file does not exist yet it says so and writes nothing.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_llm_eval.config import (
    EXPERIMENTS_DIR,
    GOLD_PATH,
    GOLD_TARGET,
    RESULTS_DIR,
    SAMPLE_PATH,
    STORE_PATH,
)
from review_llm_eval.contract import ALERTS, BASE
from review_llm_eval.data import Review
from review_llm_eval.jsonl import iter_jsonl, read_jsonl
from review_llm_eval.metrics import PRF, accuracy, binary_prf
from review_llm_eval.postprocess import expand, hotel_veto, normalize_name, veto_for
from review_llm_eval.report import markdown_table, write_csv
from review_llm_eval.runner import load_records

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class HumanEval:
    n: int
    topics: PRF  # topic mentioned or not
    pairs: PRF  # (topic, polarity)
    staff: PRF
    incident_accuracy: float
    alerts: dict[str, PRF]
    recommends_accuracy: float


def load_gold(path: Path) -> dict[int, Row]:
    if not path.exists():
        return {}
    return {row["review_id"]: row for row in iter_jsonl(path)}


def _prf(pred: set[Any], gold: set[Any]) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


def evaluate_outputs(
    gold: Mapping[int, Row], outputs: Mapping[int, dict[str, Any] | None]
) -> HumanEval:
    """``outputs`` are expanded model outputs by review id (``None`` = the model failed,
    counted as an empty answer: a failure is a real cost, it is not dropped)."""
    empty: dict[str, Any] = {
        "opinions": [],
        "staff": [],
        "incident": None,
        "recommends": False,
        **dict.fromkeys(ALERTS, False),
    }
    ids = sorted(gold)
    t = p = s = (0, 0, 0)
    alert_counts = {a: (0, 0, 0) for a in ALERTS}
    incident_hits: list[bool] = []
    recommends_hits: list[bool] = []
    for rid in ids:
        g, o = gold[rid], outputs.get(rid) or empty
        g_pairs = {(x["topic"], x["polarity"]) for x in g["opinions"]}
        o_pairs = {(x["topic"], x["polarity"]) for x in o["opinions"]}
        t = tuple(
            map(sum, zip(t, _prf({x for x, _ in o_pairs}, {x for x, _ in g_pairs}), strict=True))
        )
        p = tuple(map(sum, zip(p, _prf(o_pairs, g_pairs), strict=True)))
        g_staff = {normalize_name(n) for n in g["staff"]}
        o_staff = {normalize_name(n) for n in o["staff"]}
        s = tuple(map(sum, zip(s, _prf(o_staff, g_staff), strict=True)))
        for a in ALERTS:
            hit = _prf({a} if o[a] else set(), {a} if a in g["alerts"] else set())
            alert_counts[a] = tuple(map(sum, zip(alert_counts[a], hit, strict=True)))
        incident_hits.append(o["incident"] == g["incident"])
        recommends_hits.append(bool(o["recommends"]) == bool(g["recommends"]))
    return HumanEval(
        n=len(ids),
        topics=binary_prf(*t),
        pairs=binary_prf(*p),
        staff=binary_prf(*s),
        incident_accuracy=accuracy(incident_hits, [True] * len(incident_hits)),
        alerts={a: binary_prf(*c) for a, c in alert_counts.items()},
        recommends_accuracy=accuracy(recommends_hits, [True] * len(recommends_hits)),
    )


def layer1_accuracy(gold: Mapping[int, Row], store_path: Path) -> tuple[int, float]:
    if not store_path.exists():
        return 0, float("nan")
    conn = sqlite3.connect(store_path)
    rows = dict(conn.execute("SELECT review_id, sentiment FROM review_enrichment").fetchall())
    ids = [i for i in gold if i in rows]
    return len(ids), accuracy([gold[i]["sentiment"] for i in ids], [rows[i] for i in ids])


def pending_message(n_labelled: int, target: int = GOLD_TARGET) -> str:
    return f"Human-labelled evaluation: pending ({n_labelled}/{target} reviews labelled)"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["e1_4b", "e1_8b"])
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--store", type=Path, default=STORE_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--min-labels", type=int, default=GOLD_TARGET)
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    if len(gold) < args.min_labels:
        print(pending_message(len(gold)))
        if not gold:
            return
        print("(scoring the partial gold set anyway; numbers are provisional)")

    reviews = {r["review_id"]: Review.from_dict(r) for r in read_jsonl(args.sample)}
    base = hotel_veto({r.hotel for r in reviews.values()})
    headers = [
        "run", "n", "topic_P", "topic_R", "topic_F1", "pair_P", "pair_R", "pair_F1",
        "staff_P", "staff_R", "incident_acc", "recommends_acc",
    ]  # fmt: skip
    rows: list[list[Any]] = []
    alert_rows: list[list[Any]] = []
    for run_id in args.runs:
        outputs: dict[int, dict[str, Any] | None] = {}
        for rec in load_records(args.experiments, run_id):
            out = (rec.get("extraction") or {}).get("output")
            rv = reviews.get(rec["review_id"])
            if out is not None and rv is not None:
                outputs[rec["review_id"]] = expand(
                    out, BASE, rv.full_text, veto_for(rv.hotel, base)
                )
        ev = evaluate_outputs(gold, outputs)
        rows.append(
            [
                run_id, ev.n, ev.topics.precision, ev.topics.recall, ev.topics.f1,
                ev.pairs.precision, ev.pairs.recall, ev.pairs.f1, ev.staff.precision,
                ev.staff.recall, ev.incident_accuracy, ev.recommends_accuracy,
            ]
        )  # fmt: skip
        for alert, prf in ev.alerts.items():
            alert_rows.append([run_id, alert, prf.support, prf.precision, prf.recall])
    n1, acc1 = layer1_accuracy(gold, args.store)
    write_csv(args.results / "human_eval_summary.csv", headers, rows)
    write_csv(
        args.results / "human_eval_alerts.csv", ["run", "alert", "gold", "P", "R"], alert_rows
    )
    print(markdown_table(headers, rows))
    print()
    print(markdown_table(["run", "alert", "gold", "P", "R"], alert_rows))
    print(f"\nlayer 1 sentiment accuracy vs human: {acc1:.3f} on {n1} reviews")


if __name__ == "__main__":
    main()
