"""Two small side experiments on tagged runs produced by ``review_llm_eval.run``.

stability  Re-run the same reviews with identical settings (``--tag rerun``) and
           measure how often the answer changes. Temperature 0 + fixed seed is
           expected to be deterministic; this checks it instead of assuming it.
optional   Run with ``--schema-variant optional --tag optional`` (``complaint`` and
           ``would_return`` not in ``required``) and count how often the model
           emits those keys at all.

Usage:
    python -m review_llm_eval.run --model gemma3:4b --limit 50 --tag rerun
    python -m review_llm_eval.experiments stability --models gemma3:4b
    python -m review_llm_eval.run --model gemma3:4b --limit 50 --tag optional \\
        --schema-variant optional
    python -m review_llm_eval.experiments optional --models gemma3:4b

``--base-tag``/``--tag`` compare any two tagged runs (e.g. two sequential runs).
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from review_llm_eval.config import ASPECTS, DEFAULT_MODELS, OUTPUTS_DIR, RESULTS_DIR
from review_llm_eval.jsonl import iter_jsonl
from review_llm_eval.metrics import mean
from review_llm_eval.report import markdown_table, write_csv
from review_llm_eval.run import output_path

Row = dict[str, Any]


def _load(path: Path) -> dict[int, Row]:
    return {row["review_id"]: row for row in iter_jsonl(path)} if path.exists() else {}


def stability(base: dict[int, Row], rerun: dict[int, Row]) -> dict[str, float | int]:
    """Share of reviews whose answer is identical in two runs with the same settings."""
    ids = sorted(i for i in rerun if i in base and base[i]["status"] == rerun[i]["status"] == "ok")
    pairs = [(base[i], rerun[i]) for i in ids]
    outs = [(a["output"], b["output"]) for a, b in pairs]
    return {
        "n": len(ids),
        "raw_text_identical": mean([a["raw"] == b["raw"] for a, b in pairs]),
        "parsed_output_identical": mean([a == b for a, b in outs]),
        "sentiment_identical": mean([a["sentiment"] == b["sentiment"] for a, b in outs]),
        "aspect_values_identical": mean([a["aspects"] == b["aspects"] for a, b in outs]),
        "aspect_cells_identical": mean(
            [a["aspects"][k] == b["aspects"][k] for a, b in outs for k in ASPECTS]
        ),
        "would_return_identical": mean([a["would_return"] == b["would_return"] for a, b in outs]),
        "complaint_identical": mean([a["complaint"] == b["complaint"] for a, b in outs]),
    }


def optional_presence(
    optional: dict[int, Row], base: dict[int, Row]
) -> dict[str, float | int | str]:
    """How often optional keys appear, next to how often the required variant (same
    reviews) filled them with a non-null value."""
    ids = sorted(i for i, r in optional.items() if r["status"] == "ok")
    opt = [optional[i]["output"] for i in ids]
    req = [base[i]["output"] for i in ids if i in base and base[i]["status"] == "ok"]
    paired = [
        (optional[i]["output"], base[i]["output"])
        for i in ids
        if i in base and base[i]["status"] == "ok"
    ]
    # key order of the generated JSON (dicts keep insertion order when parsed)
    orders = Counter(",".join(o) for o in opt)
    top_order, top_count = orders.most_common(1)[0] if orders else ("", 0)
    return {
        "n": len(ids),
        "optional_valid_rate": len(ids) / len(optional) if optional else float("nan"),
        "complaint_key_present": mean([("complaint" in o) for o in opt]),
        "would_return_key_present": mean([("would_return" in o) for o in opt]),
        "optional_variant_complaint_non_null": mean([o.get("complaint") is not None for o in opt]),
        "required_variant_complaint_non_null": mean([o["complaint"] is not None for o in req]),
        "optional_variant_would_return_non_null": mean(
            [o.get("would_return") is not None for o in opt]
        ),
        "required_variant_would_return_non_null": mean(
            [o["would_return"] is not None for o in req]
        ),
        "sentiment_same_as_required_variant": mean(
            [o["sentiment"] == r["sentiment"] for o, r in paired]
        ),
        "most_common_key_order": top_order.replace(",", " > "),
        "most_common_key_order_share": top_count / len(opt) if opt else float("nan"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["stability", "optional"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--base-tag", default=None, help="stability: first run (default: main)")
    parser.add_argument("--tag", default=None, help="stability: 'rerun'; optional: 'optional'")
    parser.add_argument("--out-name", default=None, help="CSV file name in results/")
    args = parser.parse_args(argv)
    tag = args.tag or ("rerun" if args.experiment == "stability" else "optional")

    rows: list[list[Any]] = []
    headers: list[str] = []
    for model in args.models:
        base = _load(output_path(model, args.base_tag, args.outputs))
        other = _load(output_path(model, tag, args.outputs))
        if args.experiment == "stability":
            stats = stability(base, other)
        else:
            stats = optional_presence(other, base)
        headers = ["model", "runs_compared", *stats]
        rows.append([model, f"{args.base_tag or 'main'} vs {tag}", *stats.values()])
    default_name = {"stability": "stability.csv", "optional": "optional_fields.csv"}
    name = args.out_name or default_name[args.experiment]
    write_csv(args.results / name, headers, rows)
    print(markdown_table(headers, rows))


if __name__ == "__main__":
    main()
