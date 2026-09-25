"""Pure functions that turn experiment records into numbers. No I/O, no network."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from review_llm_eval.contract import BOOLEAN_FIELDS, Variant
from review_llm_eval.metrics import mean, percentile
from review_llm_eval.postprocess import expand, fold, is_filler, quote_support

Record = dict[str, Any]


def _attempts(rec: Record) -> list[dict[str, Any]]:
    ext = rec.get("extraction")
    return list(ext["attempts"]) if ext else []


def output_of(rec: Record) -> dict[str, Any] | None:
    ext = rec.get("extraction")
    return ext["output"] if ext else None


def run_summary(records: Sequence[Record], session: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reliability and speed of one run."""
    n = len(records)
    firsts = [a[0] for r in records if (a := _attempts(r))]
    all_attempts = [a for r in records for a in _attempts(r)]
    valid = [r for r in records if output_of(r) is not None]
    retried = [r for r in records if len(_attempts(r)) > 1]
    rescued = [r for r in retried if output_of(r) is not None]
    eval_counts = [a["eval_count"] for a in firsts if a.get("eval_count") is not None]
    gen_tokens = sum(a.get("eval_count") or 0 for a in all_attempts)
    gen_seconds = sum(a.get("eval_duration_s") or 0 for a in all_attempts)
    latencies = [sum(a["latency_s"] for a in _attempts(r)) for r in records if _attempts(r)]
    wall = float(session["wall_s"]) if session else float("nan")
    done = int(session["n_done"]) if session else n
    return {
        "n": n,
        "valid": len(valid),
        "valid_first_attempt": sum(a["error"] is None for a in firsts),
        "retried": len(retried),
        "rescued_by_retry": len(rescued),
        "failed": n - len(valid) - sum(r.get("network_error") is not None for r in records),
        "network_errors": sum(r.get("network_error") is not None for r in records),
        "runaway_reviews": sum(
            any(a.get("done_reason") == "length" for a in _attempts(r)) for r in records
        ),
        "output_tokens_mean": mean(eval_counts),
        "output_tokens_p50": percentile(eval_counts, 50),
        "output_tokens_p95": percentile(eval_counts, 95),
        "output_tokens_max": max(eval_counts) if eval_counts else float("nan"),
        "prompt_tokens_mean": mean(
            [a["prompt_eval_count"] for a in firsts if a.get("prompt_eval_count") is not None]
        ),
        "prompt_eval_s_mean": mean(
            [a["prompt_eval_duration_s"] for a in firsts if a.get("prompt_eval_duration_s")]
        ),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "decode_tok_per_s": gen_tokens / gen_seconds if gen_seconds else float("nan"),
        "wall_s": wall,
        "reviews_per_min": done / wall * 60 if wall else float("nan"),
        "aggregate_tok_per_s": gen_tokens / wall if wall else float("nan"),
    }


# --- comparing two runs field by field ----------------------------------------------------

FIELD_VIEWS = (
    "json",
    "opinions",
    "quotes",
    "staff",
    "incident",
    *BOOLEAN_FIELDS,
    "nights",
    "language",
    "summary",
)


def comparable(raw: Mapping[str, Any], variant: Variant) -> dict[str, Any]:
    """A variant's raw output reduced to long names, without filtering (no source text),
    so outputs of different key styles can be compared."""
    e = expand(dict(raw), variant)
    return {
        "opinions": frozenset((o["topic"], o["polarity"]) for o in e["opinions"]),
        "quotes": tuple(sorted(fold(o["quote"] or "") for o in e["opinions"])),
        "staff": frozenset(fold(s) for s in e["staff"]),
        "incident": e["incident"],
        **{b: e[b] for b in BOOLEAN_FIELDS},
        "nights": e["nights"],
        "language": e["model_language"],
        "summary": e["summary"],
    }


def agreement(
    a: Sequence[Record],
    b: Sequence[Record],
    variant_a: Variant,
    variant_b: Variant,
) -> dict[str, float | int]:
    """Share of reviews (valid in both runs) with an identical value, per field."""
    by_id = {r["review_id"]: output_of(r) for r in b}
    pairs = [
        (out_a, by_id[r["review_id"]])
        for r in a
        if (out_a := output_of(r)) is not None and by_id.get(r["review_id"]) is not None
    ]
    result: dict[str, float | int] = {"n_pairs": len(pairs)}
    if not pairs:
        return result
    same_contract = variant_a.keys == variant_b.keys and variant_a.incident == variant_b.incident
    result["json"] = (
        mean([json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in pairs])
        if same_contract
        else float("nan")
    )
    views = [(comparable(x, variant_a), comparable(y, variant_b)) for x, y in pairs]
    for name in FIELD_VIEWS[1:]:
        result[name] = mean([va[name] == vb[name] for va, vb in views])
    return result


# --- optional keys (E3) --------------------------------------------------------------------


def key_presence(records: Sequence[Record], key: str) -> dict[str, float | int]:
    outputs = [o for r in records if (o := output_of(r)) is not None]
    present = [o for o in outputs if key in o]
    return {
        "n_valid": len(outputs),
        "key_present": len(present) / len(outputs) if outputs else float("nan"),
        "non_null_when_present": (
            sum(o[key] is not None for o in present) / len(present) if present else float("nan")
        ),
        "non_null_overall": (
            sum(o.get(key) is not None for o in outputs) / len(outputs) if outputs else float("nan")
        ),
    }


# --- summary length (E5) -------------------------------------------------------------------

_ECHO = re.compile(r"palabra|\bmaximo\b|\bmaximum\b|\bwords?\b")
"""The summary talks about its own length ("15 palabras", "máximo"). Bare numbers are not
counted: "esperó 45 minutos" is content, not an echo."""


def summary_stats(
    records: Sequence[Record], variant: Variant, limit_for: Callable[[int | None], int] | None
) -> dict[str, Any]:
    """Echo and runaways (an attempt stopped at num_predict) in the summaries of a run;
    with ``limit_for`` (rating -> word limit) also how many summaries exceed it."""
    k = variant.key
    valid = [(r, o) for r in records if (o := output_of(r)) is not None]
    summaries = [str(o.get(k("summary")) or "") for _, o in valid]
    first_attempts = [a[0] for r in records if (a := _attempts(r))]
    raw_firsts = [a["raw"] for a in first_attempts]

    def echoes(text: str) -> bool:
        return bool(_ECHO.search(fold(text)))

    words = [len(s.split()) for s in summaries]
    over = (
        sum(n > limit_for(r.get("rating")) for n, (r, _) in zip(words, valid, strict=True))
        if limit_for is not None
        else None
    )
    return {
        "n": len(records),
        "valid": len(valid),
        "runaway_first_attempt": sum(a.get("done_reason") == "length" for a in first_attempts),
        "echo_in_valid_summaries": sum(echoes(s) for s in summaries),
        "echo_in_first_raw": sum(echoes(_summary_from_raw(r, k("summary"))) for r in raw_firsts),
        "summary_words_mean": mean(words),
        "summary_words_p95": percentile(words, 95),
        "summary_words_max": max(words) if words else float("nan"),
        "over_limit": over,
    }


def _summary_from_raw(raw: str, key: str) -> str:
    """The summary text of a raw answer, also when the JSON was cut off."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)', raw or "")
    return match.group(1) if match else ""


# --- quotes and names ------------------------------------------------------------------------


def support_histogram(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, int]]:
    """Distribution of ``quote_support`` over (quote, source) pairs, fillers excluded."""
    bins = Counter()
    for quote, source in pairs:
        if is_filler(quote):
            continue
        s = quote_support(quote, source)
        label = (
            "1.0 (literal)"
            if s == 1.0
            else f"[{int(s * 10) / 10:.1f}, {int(s * 10) / 10 + 0.1:.1f})"
        )
        bins[label] += 1
    order = [f"[{i / 10:.1f}, {i / 10 + 0.1:.1f})" for i in range(10)] + ["1.0 (literal)"]
    return [(label, bins.get(label, 0)) for label in order]


# --- how often each flag fires --------------------------------------------------------------


def flag_counts(
    records: Sequence[Record], variant: Variant, reviews: Mapping[int, Any]
) -> dict[str, int]:
    """Per run: how many valid outputs raise each alert (and how many of those have no
    supporting phrase in the text), qualifiers, incidents, opinions and staff names."""
    from review_llm_eval.cues import has_cue

    counts: Counter[str] = Counter()
    for rec in records:
        out = output_of(rec)
        if out is None:
            continue
        review = reviews[rec["review_id"]]
        e = expand(out, variant, review.full_text)
        counts["valid"] += 1
        for name in ("says_no_return", "legal_action", "theft", "illness", "bad_faith"):
            if e[name]:
                counts[name] += 1
                counts[f"{name}_no_phrase"] += not has_cue(name, review.full_text)
        for name in ("returning_guest", "with_children", "recommends"):
            counts[name] += bool(e[name])
        incident = e["incident"]
        counts["incident"] += bool(incident) if isinstance(incident, list) else incident is not None
        counts["opinions"] += len(e["opinions"])
        counts["staff_names"] += len(e["staff"])
    return dict(counts)


def alert_by_rating(
    records: Sequence[Record], variant: Variant, reviews: Mapping[int, Any], alert: str
) -> dict[int, tuple[int, int, int]]:
    """Per star rating: (valid outputs, outputs raising ``alert``, of those how many have
    no supporting phrase in the text). Shows whether an alert follows the text or the
    stars."""
    from review_llm_eval.cues import has_cue

    counts: dict[int, list[int]] = {}
    for rec in records:
        out = output_of(rec)
        review = reviews[rec["review_id"]]
        if out is None or review.rating is None:
            continue
        row = counts.setdefault(review.rating, [0, 0, 0])
        row[0] += 1
        if out.get(variant.key(alert)):
            row[1] += 1
            row[2] += not has_cue(alert, review.full_text)
    return {rating: (v[0], v[1], v[2]) for rating, v in sorted(counts.items())}


FLAG_COLUMNS = (
    "valid",
    "says_no_return",
    "says_no_return_no_phrase",
    "bad_faith",
    "bad_faith_no_phrase",
    "theft",
    "illness",
    "legal_action",
    "recommends",
    "returning_guest",
    "with_children",
    "incident",
    "opinions",
    "staff_names",
)
