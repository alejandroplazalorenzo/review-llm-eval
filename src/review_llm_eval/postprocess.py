"""Deterministic post-processing of a validated model output.

Everything that can be decided without judgement is done here, in code, not asked of
the model (asking costs output tokens on every call and guarantees nothing):

- short keys and codes are expanded to the long names stored in the database;
- repeated (topic, polarity) pairs are collapsed to the first one;
- filler quotes ("no se menciona", "n/a") and quotes that do not come from the review
  are set to ``None`` (the opinion itself is kept: the label can be right while the
  transcription is wrong);
- staff names must be written in the review itself, must not be a job title or the
  name of a hotel, chain or place, and are de-duplicated;
- the language comes from the deterministic detector, not from the model (the model's
  answer stays in the raw output).

``QUOTE_MIN_SUPPORT`` was set by looking at the distribution of the model's quotes on the
public data, and the filler pattern from the placeholders that actually appeared there
(see their docstrings).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

from review_llm_eval.contract import (
    BOOLEAN_FIELDS,
    INCIDENTS,
    LANGUAGES,
    POLARITIES,
    TOPICS,
    JsonObject,
    Variant,
)

QUOTE_MIN_SUPPORT = 0.5
"""Minimum share of a quote's content words (4+ letters) that must appear in the review.
Placed in the empty gap of the observed distribution: over the 1,536 quotes of the E1
runs (qwen3:4b and qwen3:8b, 200 reviews each) no quote has a support in [0.4, 0.6);
below the gap sit 7 quotes that copy the topic definitions of the prompt; above it,
transcriptions where the model fixed the guest's typos, plus two short paraphrases at
0.67 (``results/quote_support.md``)."""

FILLER_QUOTE = re.compile(
    r"^(?:[-.\s]*|n/?a|none|null|sin cita"
    r"|(?:no (?:se )?)?(?:menciona|especifica|indica)(?: nada)?)$"
)
"""A placeholder written instead of a quote (matched on the folded quote, final dot
removed). In every run of this repository the only one that appeared was "no se menciona"
(twice, qwen3:8b in E1); the other alternatives are its obvious variants and the usual
"not applicable" tokens."""

JOB_TITLES = frozenset(
    {
        "animador",
        "animadora",
        "animadores",
        "bartender",
        "barman",
        "botones",
        "butler",
        "camarero",
        "camarera",
        "camareros",
        "chef",
        "cocinero",
        "cocinera",
        "concierge",
        "conserje",
        "director",
        "directora",
        "equipo",
        "gerente",
        "guia",
        "jefe",
        "mayordomo",
        "mayordoma",
        "mesero",
        "mesera",
        "meseros",
        "nadie",
        "personal",
        "recepcionista",
        "recepcionistas",
        "senor",
        "senora",
        "socorrista",
        "staff",
        "todos",
        "supervisor",
        "supervisora",
        "manager",
    }
)
"""Roles and generic words that are not a person's name (folded, articles removed)."""

PLACE_WORDS = frozenset(
    {"punta cana", "bavaro", "cap cana", "republica dominicana", "dominicana", "santo domingo"}
)

_ARTICLES = re.compile(r"^(el|la|los|las|un|una|nuestro|nuestra|mi)\s+")
_WORD = re.compile(r"\w+")


def fold(text: str | None) -> str:
    """Lower case, no diacritics, whitespace collapsed."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.split())


def normalize_name(name: str) -> str:
    """Identity form of a staff name: identity is (normalized name, hotel)."""
    return fold(name)


def quote_support(quote: str, source: str) -> float:
    """Share of the quote that comes from ``source`` (1.0 = literal).

    Literal after folding (case, accents, spacing) counts as 1.0; otherwise the share of
    the quote's content words (4+ letters) present in the source. A quote with no
    content word is only accepted when literal.
    """
    q, s = fold(quote), fold(source)
    if not q:
        return 0.0
    if q in s or q.replace(" ", "") in s.replace(" ", ""):
        return 1.0
    words = [w for w in _WORD.findall(q) if len(w) >= 4]
    if not words:
        return 0.0
    source_words = set(_WORD.findall(s))
    return sum(w in source_words for w in words) / len(words)


def is_literal(quote: str, source: str) -> bool:
    return quote_support(quote, source) == 1.0


def is_filler(quote: str | None) -> bool:
    return FILLER_QUOTE.match(fold(quote).rstrip(".").strip()) is not None


def hotel_veto(hotel_names: Iterable[str]) -> frozenset[str]:
    """Names the model must not return as staff, built from the dataset metadata at run
    time (so it follows the data instead of a hand-kept list): every full hotel name,
    every word shared by two or more hotel names (chains and brands: "iberostar",
    "riu", "dreams"...) and a few place names. A candidate is vetoed only when it is
    exactly one of these."""
    veto = set(PLACE_WORDS)
    hotels_per_word: dict[str, set[str]] = {}
    for name in hotel_names:
        folded = fold(name)
        if not folded:
            continue
        veto.add(folded)
        for word in _WORD.findall(folded):
            if len(word) >= 3:
                hotels_per_word.setdefault(word, set()).add(folded)
    veto.update(word for word, hotels in hotels_per_word.items() if len(hotels) >= 2)
    return frozenset(veto)


def veto_for(hotel: str, base: frozenset[str]) -> frozenset[str]:
    """``base`` plus the words of the review's own hotel name. Guests call a hotel by its
    distinctive word ("el Esmeralda"); a name equal to it is ambiguous and is dropped:
    a wrong name in a per-employee count does more harm than a missing one."""
    return base | frozenset(w for w in _WORD.findall(fold(hotel)) if len(w) >= 3)


def is_role_or_generic(name: str) -> bool:
    folded = _ARTICLES.sub("", fold(name))
    return folded in JOB_TITLES or all(w in JOB_TITLES for w in folded.split())


def name_in_text(name: str, source: str) -> bool:
    """Whole-word match after folding ("Ana" is not in "banana")."""
    folded = fold(name)
    if not folded:
        return False
    return re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", fold(source)) is not None


def clean_staff(
    names: Sequence[Any] | None, source: str | None, veto: frozenset[str] = frozenset()
) -> list[str]:
    """Drop roles, hotel/place names, names not written in ``source`` and repeats.

    Without ``source`` the grounding check is skipped (nothing to check against)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in names or []:
        name = " ".join(str(raw).split())
        key = normalize_name(name)
        if not key or key in seen or key in veto or is_role_or_generic(name):
            continue
        if source is not None and not name_in_text(name, source):
            continue
        seen.add(key)
        out.append(name)
    return out


def _decode(value: Any, table: dict[str, Any], variant: Variant) -> Any:
    """Short code -> long name (long variants already carry the long name)."""
    if variant.keys == "long":
        return value
    return table.get(value)


def expand(
    raw: JsonObject,
    variant: Variant,
    source: str | None = None,
    veto: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Model output -> dict with the long names of the database columns.

    ``source`` is the review text (title + body) used to check quotes and names; without
    it those two checks are skipped. The raw output is returned too, so a change in this
    mapping can be re-applied later without calling the model again.
    """
    k = variant.key
    topics = {c: long for c, (long, _) in TOPICS.items()}
    opinions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw.get(k("opinions")) or []:
        topic = _decode(item.get(k("topic")), topics, variant)
        polarity = _decode(item.get(k("polarity")), POLARITIES, variant)
        if not topic or not polarity or (topic, polarity) in seen:
            continue
        seen.add((topic, polarity))
        quote = " ".join(str(item.get(k("quote")) or "").split())
        if is_filler(quote) or (
            source is not None and quote_support(quote, source) < QUOTE_MIN_SUPPORT
        ):
            quote = ""
        opinions.append({"topic": topic, "polarity": polarity, "quote": quote or None})

    incident_raw = raw.get(k("incident"))
    if variant.incident == "list":
        decoded = (_decode(c, _INCIDENT_LONG, variant) for c in incident_raw or [])
        incident: Any = sorted({c for c in decoded if c})
    elif variant.incident == "free_text":
        incident = incident_raw
    else:
        incident = _decode(incident_raw, _INCIDENT_LONG, variant)
        incident = None if incident in (None, "none") else incident

    language = _decode(raw.get(k("language")), LANGUAGES, variant)
    nights = raw.get(k("nights"))
    summary = " ".join(str(raw.get(k("summary")) or "").split())
    return {
        "opinions": opinions,
        "staff": clean_staff(raw.get(k("staff")), source, veto),
        "incident": incident,
        **{name: bool(raw.get(k(name))) for name in BOOLEAN_FIELDS},
        "nights": nights if isinstance(nights, int) and nights > 0 else None,
        "model_language": language,
        "summary": summary or None,
        "raw_output": raw,
    }


_INCIDENT_LONG: dict[str, Any] = {code: long for code, (long, _) in INCIDENTS.items()}
