"""The ten objective checks used to compare models (experiment E1). They were fixed
after the qwen3:4b run and before the qwen3:8b run, and not changed afterwards: they are
not blind to the 4b's answers. None needs a gold label: each one measures a behaviour
that is wrong by construction (an invalid answer, a quote that is not in the text, a
name that is a job title...), not whether a judgement is right.

All of them are computed on the model's raw output, before ``postprocess`` repairs it:
they measure the model, not the pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from review_llm_eval.contract import BASE, BOOLEAN_FIELDS, Variant
from review_llm_eval.cues import has_cue
from review_llm_eval.data import Review
from review_llm_eval.postprocess import (
    fold,
    is_filler,
    is_role_or_generic,
    name_in_text,
    quote_support,
)

Record = dict[str, Any]

UNSUPPORTED_BELOW = 0.5
"""A quote with less than half of its content words in the review counts as unsupported.
Fixed here as a round number, independently of the pipeline threshold
(``postprocess.QUOTE_MIN_SUPPORT``), which is derived from the data."""


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    description: str
    higher_is_better: bool


CHECKS: tuple[Check, ...] = (
    Check("valid_first_attempt", "share of reviews with a schema-valid first answer", True),
    Check("runaway", "share of reviews with an attempt cut at num_predict", False),
    Check("duplicate_pairs", "share of valid outputs repeating a (topic, polarity) pair", False),
    Check("name_not_in_text", "share of staff names not written in the review", False),
    Check("role_as_name", "staff entries that are a job title or generic word (count)", False),
    Check("unsupported_quote", "share of quotes below the support threshold", False),
    Check("filler_quote", "share of quotes that are placeholders", False),
    Check(
        "no_return_5star_no_cue",
        "5-star reviews flagged 'will not return' without any such phrase (count)",
        False,
    ),
    Check("legal_no_cue", "reviews flagged 'legal action' without any such phrase (count)", False),
    Check("never_varies", "output fields with one single value over the sample (count)", False),
)

VARYING_FIELDS: tuple[str, ...] = ("incident", *BOOLEAN_FIELDS, "nights", "language")


def _share(num: int, den: int) -> float:
    return num / den if den else float("nan")


def compute(
    records: Sequence[Record], reviews: Mapping[int, Review], variant: Variant = BASE
) -> dict[str, float | int]:
    """Values of every check over ``records`` (one per review, as written by the runner)."""
    k = variant.key
    n = len(records)
    first_ok = runaway = 0
    outputs: list[tuple[Review, dict[str, Any]]] = []
    for rec in records:
        ext = rec.get("extraction")
        if not ext:
            continue
        attempts = ext["attempts"]
        first_ok += bool(attempts) and attempts[0]["error"] is None
        runaway += any(a.get("done_reason") == "length" for a in attempts)
        if ext["output"] is not None:
            outputs.append((reviews[rec["review_id"]], ext["output"]))

    dup = 0
    names = names_missing = roles = 0
    quotes = unsupported = fillers = 0
    no_return_5 = legal_no_cue = 0
    values: dict[str, set[Any]] = {f: set() for f in VARYING_FIELDS}
    for review, out in outputs:
        source = review.full_text
        pairs = [(o.get(k("topic")), o.get(k("polarity"))) for o in out.get(k("opinions"), [])]
        dup += len(pairs) != len(set(pairs))
        for op in out.get(k("opinions"), []):
            quote = str(op.get(k("quote")) or "")
            quotes += 1
            if is_filler(quote):
                fillers += 1
            elif quote_support(quote, source) < UNSUPPORTED_BELOW:
                unsupported += 1
        for raw_name in out.get(k("staff"), []):
            name = " ".join(str(raw_name).split())
            if is_role_or_generic(name) or not fold(name):
                roles += 1
                continue
            names += 1
            names_missing += not name_in_text(name, source)
        if review.rating == 5 and out.get(k("says_no_return")):
            no_return_5 += not has_cue("says_no_return", source)
        if out.get(k("legal_action")):
            legal_no_cue += not has_cue("legal_action", source)
        for f in VARYING_FIELDS:
            value = out.get(k(f))
            if f == "nights":
                value = value is not None
            values[f].add(value if not isinstance(value, list) else tuple(value))

    return {
        "n_reviews": n,
        "n_valid": len(outputs),
        "valid_first_attempt": _share(first_ok, n),
        "runaway": _share(runaway, n),
        "duplicate_pairs": _share(dup, len(outputs)),
        "name_not_in_text": _share(names_missing, names),
        "role_as_name": roles,
        "unsupported_quote": _share(unsupported, quotes - fillers),
        "filler_quote": _share(fillers, quotes),
        "no_return_5star_no_cue": no_return_5,
        "legal_no_cue": legal_no_cue,
        "never_varies": sum(len(v) <= 1 for v in values.values()),
        "n_names": names,
        "n_names_missing": names_missing,
        "n_quotes": quotes,
        "n_fillers": fillers,
        "n_unsupported": unsupported,
        "fields_never_varying": ",".join(f for f, v in values.items() if len(v) <= 1),
    }


def compare(a: Mapping[str, float | int], b: Mapping[str, float | int]) -> list[str]:
    """Per check: 'a', 'b' or 'tie' (which model is better)."""
    verdicts = []
    for check in CHECKS:
        x, y = float(a[check.name]), float(b[check.name])
        if math.isnan(x) or math.isnan(y):
            verdicts.append("n/a")
        elif abs(x - y) < 1e-12:
            verdicts.append("tie")
        elif (x > y) == check.higher_is_better:
            verdicts.append("a")
        else:
            verdicts.append("b")
    return verdicts
