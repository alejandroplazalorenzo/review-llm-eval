"""Deterministic language detection (layer 1), no dependencies.

The language is not left to the model: it is cheap to detect, it must not depend on a
GPU being available, and a model can answer "es" for a review written in English. The
model is still asked (its answer stays in the raw output) so both can be compared.

Method: count function words that are distinctive of each language (words shared with
Spanish, like "de", "la" or "no", are left out) and pick a foreign language only when it
clearly beats Spanish. Short texts with no function word default to Spanish, the
majority class, where a mistake costs least.

Calibration on the 33,038 de-duplicated reviews: the rule marks 36 as English, 1 as
Portuguese and 1 as German. 11 of them contain no Spanish function word at all; most of
the rest are bilingual (Spanish plus a translation) and a few are short Spanish reviews
with several English words, where "the dominant language" is itself debatable
(``results/l11_language.md`` compares them with the model's answer).
"""

from __future__ import annotations

import re

DETECTOR_VERSION = "langdet-v1"


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


_VOCABULARY: dict[str, frozenset[str]] = {
    "es": _words(
        """el los las del muy pero con una fue estaba habitación habitaciones también son
        hay más personal playa bien estuvimos había mucho siempre gracias atención niños
        mejor"""
    ),
    "en": _words(
        """the and was were with very we our they this that is are not but have had for
        you it there would"""
    ),
    "pt": _words(
        """não muito uma com foi você são nós eram ótimo então quarto praia funcionários
        atendimento"""
    ),
    "fr": _words("très nous avec était pour pas c'est une sont tout chambre plage personnel"),
    "it": _words("molto sono della anche però questo abbiamo stato gli camera spiaggia cibo"),
    "de": _words("und der die das nicht sehr wir ist mit auch war ein zimmer"),
}

_TOKEN = re.compile(r"[a-záéíóúüñàèìòùâêîôûãõçäöß']+")

MIN_FOREIGN_HITS = 3
"""A foreign language needs at least this many distinctive words..."""
FOREIGN_OVER_SPANISH = 1.5
"""...and more than this many times the Spanish hits."""


def scores(text: str | None) -> dict[str, int]:
    """Distinctive-word hits per language."""
    tokens = _TOKEN.findall((text or "").lower())
    return {lang: sum(t in vocab for t in tokens) for lang, vocab in _VOCABULARY.items()}


def detect(text: str | None) -> str:
    """ISO 639-1 code of the review language ("es" when in doubt)."""
    hits = scores(text)
    spanish = hits.pop("es")
    lang, best = max(hits.items(), key=lambda item: item[1])
    if best >= MIN_FOREIGN_HITS and best > FOREIGN_OVER_SPANISH * spanish:
        return lang
    return "es"
