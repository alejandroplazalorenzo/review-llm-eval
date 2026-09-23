"""Aggregate raw model outputs into reliability, proxy-accuracy and agreement tables.

Usage: python -m review_llm_eval.analyze [--models qwen2.5:7b-instruct gemma3:4b]

Writes small CSV/Markdown files to ``results/`` (committed) and a list of
disagreements to ``data/analysis/`` (not committed; it contains review text).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from review_llm_eval.config import (
    ASPECTS,
    DATA_DIR,
    DEFAULT_MODELS,
    OUTPUTS_DIR,
    OVERALL_SENTIMENTS,
    PROXY_LABELS,
    RESULTS_DIR,
    SAMPLE_PATH,
)
from review_llm_eval.data import Review
from review_llm_eval.jsonl import iter_jsonl, read_jsonl, write_jsonl
from review_llm_eval.metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    mean,
    percentile,
    prf_for_label,
)
from review_llm_eval.report import markdown_table, write_csv
from review_llm_eval.run import output_path

Row = dict[str, Any]

POLAR = frozenset({"positive", "negative", "mixed"})


def proxy_label(rating: int) -> str | None:
    """Star rating -> noisy sentiment proxy. 3 stars has no proxy label."""
    if rating <= 2:
        return "negative"
    if rating >= 4:
        return "positive"
    return None


def load_results(path: Path) -> dict[int, Row]:
    """review_id -> result row (a later line for the same id wins)."""
    return {row["review_id"]: row for row in iter_jsonl(path)}


def load_runs(path: Path) -> list[Row]:
    run_file = path.with_suffix(".run.json")
    return json.loads(run_file.read_text(encoding="utf-8")) if run_file.exists() else []


# --------------------------------------------------------------------------- reliability


@dataclass(frozen=True, slots=True)
class Reliability:
    n: int
    valid_first_attempt: int
    retried: int
    recovered_by_retry: int
    failed: int
    hit_token_limit: int
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    server_mean_s: float
    output_tokens_mean: float
    prompt_tokens_mean: float
    reviews_per_min: float

    @property
    def validity_rate(self) -> float:
        return (self.n - self.failed) / self.n if self.n else float("nan")

    @property
    def first_attempt_rate(self) -> float:
        return self.valid_first_attempt / self.n if self.n else float("nan")

    def as_row(self) -> list[int | float]:
        return [
            self.n, self.validity_rate, self.first_attempt_rate, self.retried,
            self.recovered_by_retry, self.failed, self.hit_token_limit, self.latency_mean_s,
            self.latency_p50_s, self.latency_p95_s, self.server_mean_s,
            self.output_tokens_mean, self.prompt_tokens_mean, self.reviews_per_min,
        ]  # fmt: skip


def reliability(results: Sequence[Row], runs: Sequence[Row]) -> Reliability:
    first_ok = sum(r["n_attempts"] == 1 and r["status"] == "ok" for r in results)
    retried = sum(r["n_attempts"] > 1 for r in results)
    recovered = sum(r["n_attempts"] > 1 and r["status"] == "ok" for r in results)
    failed = sum(r["status"] != "ok" for r in results)
    attempts = [a for r in results for a in r["attempts"]]
    latencies = [r["latency_s"] for r in results]
    wall = sum(run["wall_clock_s"] for run in runs)
    processed = sum(run["n_reviews"] for run in runs)
    return Reliability(
        n=len(results),
        valid_first_attempt=first_ok,
        retried=retried,
        recovered_by_retry=recovered,
        failed=failed,
        hit_token_limit=sum(a.get("done_reason") == "length" for a in attempts),
        latency_mean_s=mean(latencies),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        server_mean_s=mean([a["total_duration_s"] for a in attempts if a.get("total_duration_s")]),
        output_tokens_mean=mean([a["eval_count"] for a in attempts if a.get("eval_count")]),
        prompt_tokens_mean=mean(
            [a["prompt_eval_count"] for a in attempts if a.get("prompt_eval_count")]
        ),
        reviews_per_min=processed / wall * 60 if wall else float("nan"),
    )


# ------------------------------------------------------------------------ proxy sentiment


@dataclass(frozen=True, slots=True)
class ProxyScores:
    n: int
    accuracy: float
    macro_f1: float
    f1_negative: float
    f1_positive: float
    confusion: list[list[int]]  # rows PROXY_LABELS, cols OVERALL_SENTIMENTS


def proxy_scores(results: dict[int, Row], sample: dict[int, Review]) -> ProxyScores:
    y_true: list[str] = []
    y_pred: list[str] = []
    for review_id, row in results.items():
        label = proxy_label(sample[review_id].rating)
        if label is None or row["status"] != "ok":
            continue
        y_true.append(label)
        y_pred.append(row["output"]["sentiment"])
    return ProxyScores(
        n=len(y_true),
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, PROXY_LABELS),
        f1_negative=prf_for_label(y_true, y_pred, "negative").f1,
        f1_positive=prf_for_label(y_true, y_pred, "positive").f1,
        confusion=confusion_matrix(y_true, y_pred, PROXY_LABELS, OVERALL_SENTIMENTS),
    )


def sentiment_by_rating(results: dict[int, Row], sample: dict[int, Review]) -> dict[int, Counter]:
    table: dict[int, Counter] = {k: Counter() for k in range(1, 6)}
    for review_id, row in results.items():
        if row["status"] == "ok":
            table[sample[review_id].rating][row["output"]["sentiment"]] += 1
    return table


# -------------------------------------------------------------------------- field stats


RETURN_CUE = re.compile(
    r"volv|vuelv|vuelta|regres|repet|repit|otra vez|de nuevo|nunca m[aá]s", re.IGNORECASE
)
"""Deliberately broad Spanish cues for "coming back" (volver, repetir, regresar, ...).
A review with none of them almost certainly does not state whether the reviewer
would return, so a non-null ``would_return`` there is inferred, not extracted. This
is a heuristic diagnostic, not a gold label."""


def has_return_cue(review: Review) -> bool:
    return RETURN_CUE.search(f"{review.title} {review.text}") is not None


_SPANISH_WORDS = frozenset((
    "de", "la", "el", "los", "las", "y", "en", "que", "con", "por", "para", "del", "muy",
    "una", "un", "no", "se", "su", "al", "lo", "como", "pero", "sin",
))  # fmt: skip
_ENGLISH_WORDS = frozenset((
    "the", "and", "of", "to", "in", "was", "were", "with", "for", "not", "is", "are", "a",
    "an", "no", "but", "too", "very", "they", "their",
))  # fmt: skip


def looks_spanish(text: str) -> bool:
    """Crude stop-word vote used only to check the "complaint in English" instruction
    (the schema cannot enforce a language). Ties count as English."""
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    spanish = sum(w in _SPANISH_WORDS for w in words)
    english = sum(w in _ENGLISH_WORDS for w in words)
    return spanish > english


def _n_aspects(output: Row) -> int:
    return sum(v is not None for v in output["aspects"].values())


def field_stats(results: dict[int, Row], sample: dict[int, Review]) -> dict[str, float]:
    ok = {i: r["output"] for i, r in results.items() if r["status"] == "ok"}
    if not ok:
        return {}
    outs = list(ok.values())
    no_cue = [o for i, o in ok.items() if not has_return_cue(sample[i])]
    return {
        "would_return_non_null": mean([o["would_return"] is not None for o in outs]),
        "would_return_false": mean([o["would_return"] is False for o in outs]),
        "reviews_without_return_cue": len(no_cue),
        "would_return_non_null_without_cue": mean([o["would_return"] is not None for o in no_cue]),
        "complaint_non_null": mean([o["complaint"] is not None for o in outs]),
        "complaint_looks_spanish": mean(
            [looks_spanish(o["complaint"]) for o in outs if o["complaint"] is not None]
        ),
        "aspects_per_review": mean([_n_aspects(o) for o in outs]),
        "all_aspects_non_null": mean([_n_aspects(o) == len(ASPECTS) for o in outs]),
    }


def aspect_prevalence(results: dict[int, Row]) -> dict[str, Counter]:
    table: dict[str, Counter] = {a: Counter() for a in ASPECTS}
    for row in results.values():
        if row["status"] != "ok":
            continue
        for aspect, value in row["output"]["aspects"].items():
            table[aspect][value if value is not None else "null"] += 1
    return table


# ---------------------------------------------------------------------------- agreement


@dataclass(frozen=True, slots=True)
class Agreement:
    item: str
    n: int
    kappa: float
    raw_agreement: float
    rate_a: float | None = None  # prevalence for binary items
    rate_b: float | None = None


def agreement(a: dict[int, Row], b: dict[int, Row]) -> list[Agreement]:
    """Agreement between two models on reviews both answered validly.

    This is agreement, not accuracy: two models can agree and both be wrong.
    """
    ids = sorted(i for i in a.keys() & b.keys() if a[i]["status"] == b[i]["status"] == "ok")
    out_a = [a[i]["output"] for i in ids]
    out_b = [b[i]["output"] for i in ids]
    items: list[Agreement] = []

    def add(item: str, xs: list[Any], ys: list[Any], binary: bool = False) -> None:
        raw = mean([x == y for x, y in zip(xs, ys, strict=True)])
        rate_a = mean([bool(x) for x in xs]) if binary else None
        rate_b = mean([bool(y) for y in ys]) if binary else None
        items.append(Agreement(item, len(xs), cohen_kappa(xs, ys), raw, rate_a, rate_b))

    add("sentiment (4 classes)", [o["sentiment"] for o in out_a], [o["sentiment"] for o in out_b])
    add(
        "would_return (true/false/null)",
        [str(o["would_return"]) for o in out_a],
        [str(o["would_return"]) for o in out_b],
    )
    add(
        "complaint present",
        [o["complaint"] is not None for o in out_a],
        [o["complaint"] is not None for o in out_b],
        binary=True,
    )
    for aspect in ASPECTS:
        add(
            f"aspect present: {aspect}",
            [o["aspects"][aspect] is not None for o in out_a],
            [o["aspects"][aspect] is not None for o in out_b],
            binary=True,
        )
    # Same, but only counting an opinion (positive/negative/mixed): shows how much of
    # the disagreement comes from "neutral" being used for aspects barely touched.
    for aspect in ASPECTS:
        add(
            f"aspect with opinion: {aspect}",
            [o["aspects"][aspect] in POLAR for o in out_a],
            [o["aspects"][aspect] in POLAR for o in out_b],
            binary=True,
        )
    both = [
        (x["aspects"][asp], y["aspects"][asp])
        for x, y in zip(out_a, out_b, strict=True)
        for asp in ASPECTS
        if x["aspects"][asp] is not None and y["aspects"][asp] is not None
    ]
    if both:
        add("aspect sentiment, when both mention it", [p[0] for p in both], [p[1] for p in both])
    return items


# ------------------------------------------------------------------------ disagreements


def disagreements(
    models: Sequence[str], results: dict[str, dict[int, Row]], sample: dict[int, Review]
) -> list[Row]:
    """Reviews where a model contradicts the star proxy or the models disagree."""
    rows: list[Row] = []
    for review_id, review in sample.items():
        outputs = {
            m: results[m][review_id]["output"]
            for m in models
            if review_id in results[m] and results[m][review_id]["status"] == "ok"
        }
        label = proxy_label(review.rating)
        proxy_miss = [m for m, o in outputs.items() if label and o["sentiment"] != label]
        sentiments = {o["sentiment"] for o in outputs.values()}
        if proxy_miss or len(sentiments) > 1:
            rows.append(
                {
                    **review.to_dict(),
                    "proxy": label,
                    "proxy_miss": proxy_miss,
                    "models_disagree": len(sentiments) > 1,
                    "outputs": outputs,
                }
            )
    return rows


def example_outputs(
    models: Sequence[str],
    results: dict[str, dict[int, Row]],
    sample: dict[int, Review],
    per_rating: int = 2,
) -> list[Row]:
    """A tiny illustrative extract: model outputs only, never review text or hotel.

    ``review_id`` is the row index in the public parquet file, so anyone can look
    the review up in the original dataset.
    """
    rows: list[Row] = []
    for rating in range(1, 6):
        ids = sorted(i for i, r in sample.items() if r.rating == rating)[:per_rating]
        for review_id in ids:
            rows.append(
                {
                    "review_id": review_id,
                    "stars": rating,
                    "outputs": {
                        m: results[m][review_id]["output"]
                        for m in models
                        if review_id in results[m]
                    },
                }
            )
    return rows


# --------------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    sample = {r.review_id: r for r in (Review.from_dict(x) for x in read_jsonl(args.sample))}
    results = {m: load_results(output_path(m, root=args.outputs)) for m in args.models}
    runs = {m: load_runs(output_path(m, root=args.outputs)) for m in args.models}
    res_dir: Path = args.results
    md: list[str] = [
        "# Results summary",
        "",
        f"Generated by `python -m review_llm_eval.analyze` on {date.today().isoformat()} "
        f"from {len(sample)} sampled reviews. Every number below comes from that run.",
        "",
    ]

    # Reliability
    rel = {m: reliability(list(results[m].values()), runs[m]) for m in args.models}
    rel_headers = [
        "model", "n", "valid_rate", "valid_1st_attempt_rate", "retried", "recovered",
        "failed", "hit_token_limit", "latency_mean_s", "latency_p50_s", "latency_p95_s",
        "server_mean_s", "output_tokens_mean", "prompt_tokens_mean", "reviews_per_min",
    ]  # fmt: skip
    rel_rows = [[m, *r.as_row()] for m, r in rel.items()]
    write_csv(res_dir / "reliability.csv", rel_headers, rel_rows)
    md += ["## Reliability of the structured output", "", markdown_table(rel_headers, rel_rows), ""]

    # Proxy sentiment
    proxy = {m: proxy_scores(results[m], sample) for m in args.models}
    px_headers = ["model", "n", "accuracy", "macro_f1", "f1_negative", "f1_positive"]
    px_rows = [
        [m, p.n, p.accuracy, p.macro_f1, p.f1_negative, p.f1_positive] for m, p in proxy.items()
    ]
    write_csv(res_dir / "proxy_sentiment.csv", px_headers, px_rows)
    md += [
        "## Overall sentiment vs star rating (noisy proxy, not ground truth)",
        "",
        "1-2 stars -> negative, 4-5 stars -> positive; 3-star reviews are excluded here.",
        "",
        markdown_table(px_headers, px_rows),
        "",
    ]
    conf_rows: list[list[Any]] = []
    for m, p in proxy.items():
        md += [f"Confusion matrix, {m} (rows: proxy, columns: model):", ""]
        rows = [[lab, *counts] for lab, counts in zip(PROXY_LABELS, p.confusion, strict=True)]
        md += [markdown_table(["proxy \\ model", *OVERALL_SENTIMENTS], rows), ""]
        conf_rows += [[m, *row] for row in rows]
    write_csv(res_dir / "proxy_confusion.csv", ["model", "proxy", *OVERALL_SENTIMENTS], conf_rows)

    by_rating_rows: list[list[Any]] = []
    for m in args.models:
        table = sentiment_by_rating(results[m], sample)
        for rating, counts in table.items():
            by_rating_rows.append(
                [m, rating, sum(counts.values()), *(counts[s] for s in OVERALL_SENTIMENTS)]
            )
    br_headers = ["model", "stars", "n", *OVERALL_SENTIMENTS]
    write_csv(res_dir / "sentiment_by_rating.csv", br_headers, by_rating_rows)
    md += ["Predicted sentiment by star rating (3 stars shown here):", ""]
    md += [markdown_table(br_headers, by_rating_rows), ""]

    # Field stats and aspect prevalence
    fs_headers = [
        "model", "would_return_non_null", "would_return_false", "reviews_without_return_cue",
        "would_return_non_null_without_cue", "complaint_non_null", "complaint_looks_spanish",
        "aspects_per_review", "all_aspects_non_null",
    ]  # fmt: skip
    fs_rows = []
    for m in args.models:
        stats = field_stats(results[m], sample)
        fs_rows.append([m, *(stats.get(h) for h in fs_headers[1:])])
    write_csv(res_dir / "field_stats.csv", fs_headers, fs_rows)
    md += ["## Field usage (descriptive, not accuracy)", ""]
    md += [markdown_table(fs_headers, fs_rows), ""]

    ap_rows = []
    for m in args.models:
        for aspect, counts in aspect_prevalence(results[m]).items():
            total = sum(counts.values())
            ap_rows.append(
                [m, aspect, (total - counts["null"]) / total if total else None]
                + [counts[s] for s in OVERALL_SENTIMENTS]
            )
    write_csv(
        res_dir / "aspect_prevalence.csv",
        ["model", "aspect", "mentioned_rate", *OVERALL_SENTIMENTS],
        ap_rows,
    )

    # Agreement
    ag_headers = ["model_a", "model_b", "item", "n", "kappa", "raw_agreement", "rate_a", "rate_b"]
    ag_rows = []
    for ma, mb in itertools.combinations(args.models, 2):
        for ag in agreement(results[ma], results[mb]):
            ag_rows.append(
                [ma, mb, ag.item, ag.n, ag.kappa, ag.raw_agreement, ag.rate_a, ag.rate_b]
            )
    write_csv(res_dir / "agreement.csv", ag_headers, ag_rows)
    md += [
        "## Inter-model agreement (agreement, NOT accuracy)",
        "",
        "Cohen's kappa between models. Two models can agree and both be wrong.",
        "",
        markdown_table(ag_headers, ag_rows),
        "",
    ]
    (res_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    write_jsonl(res_dir / "example_outputs.jsonl", example_outputs(args.models, results, sample))
    dis = disagreements(args.models, results, sample)
    n = write_jsonl(DATA_DIR / "analysis" / "disagreements.jsonl", dis)
    print("\n".join(md))
    print(f"\n{n} disagreement rows written to {DATA_DIR / 'analysis' / 'disagreements.jsonl'}")


if __name__ == "__main__":
    main()
