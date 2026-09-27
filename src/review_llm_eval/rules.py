"""Explicit-only fields decided in code, after the model.

L12 on the public data: the model raised "says no return" on almost every one-star review
and "explicitly recommends" on reviews that only praise the hotel, although the prompt asks
for a statement in words. Rewording the prompt does not move judgements like these (L13), so
the pipeline keeps the model's ``True`` only when the review contains a phrase that could
state it.

The phrases are the broad cues of ``cues.py``, written for the E1 checks before the qwen3:8b
run and never tuned on labels. The rule only removes flags that no phrase in the review can
support; it never adds one, so it can lower recall only where the cue list misses a way of
saying it.
"""

from __future__ import annotations

import re
from typing import Any

from review_llm_eval.cues import CUES
from review_llm_eval.postprocess import fold

RULE_VERSION = "explicit-v1"

_NEGATION = re.compile(r"\b(no|nunca|jamas|ni)\b")
# "no dudaría en recomendarlo", "no puedo más que recomendarlo": a negation that does not
# negate the recommendation.
_NOT_A_NEGATION = re.compile(r"\b(dud|dejar de|mas que|sino)")
_CLAUSE_END = re.compile(r"[.!?;]")
_LOOK_BACK = 30


def states_no_return(text: str) -> bool:
    """The review contains a phrase that could say the guest will not come back or advises
    against the hotel (broad on purpose: see ``cues.py``)."""
    return CUES["says_no_return"].search(fold(text)) is not None


def states_recommendation(text: str) -> bool:
    """The review recommends in words: a form of "recomendar" that is not negated in the same
    clause ("no lo recomiendo" is a no-return statement, not a recommendation)."""
    folded = fold(text)
    for match in CUES["recommends"].finditer(folded):
        clause = _CLAUSE_END.split(folded[: match.start()])[-1][-_LOOK_BACK:]
        negation = _NEGATION.search(clause)
        if negation is None or _NOT_A_NEGATION.search(clause[negation.end() :]):
            return True
    return False


def apply(expanded: dict[str, Any], source: str | None) -> dict[str, Any]:
    """Drop the explicit-only flags that the review text cannot support. Without the text
    the output is returned unchanged."""
    if source is None:
        return expanded
    out = dict(expanded)
    if out.get("says_no_return") and not states_no_return(source):
        out["says_no_return"] = False
    if out.get("recommends") and not states_recommendation(source):
        out["recommends"] = False
    return out
