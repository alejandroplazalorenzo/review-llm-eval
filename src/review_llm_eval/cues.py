"""Regular expressions for the phrases that an explicit-only field needs.

They are not labels and are never used to score accuracy. They are used for two
objective checks: "an alert fired on a review that contains none of the phrases that
could justify it" (E1, gates) and "a field never varies although the data contains
reviews that should trigger it" (gates). The lists are deliberately broad (e.g. "no
dudaría en volver" matches the no-return cue), so both checks *undercount* problems.

Written with the E1 checks (before the qwen3:8b run); matched on folded text (lower
case, no accents).
"""

from __future__ import annotations

import re

from review_llm_eval.postprocess import fold

_NUMBERS = "dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|catorce|quince"

CUES: dict[str, re.Pattern[str]] = {
    "says_no_return": re.compile(
        r"\b(no|nunca|jamas|ni loco)\b[^.!?]{0,40}"
        r"\b(volv|vuelv|regres|repet|repit|recomend|recomiend)"
        r"|\bnunca mas\b|\bdesaconsej"
    ),
    "legal_action": re.compile(
        r"\b(denunci|abogad|demand|juicio|juzgado|tribunal|pro ?consumidor|consumo\b"
        r"|reclamacion formal|hoja de reclamacion|policia|fiscal|legal\b|banco\b"
        r"|tarjeta de credito|contracargo|chargeback|disputa)"
    ),
    "theft": re.compile(
        r"\b(rob|hurt|desapareci|sustraj|ladron|nos quitaron|me quitaron|faltaba|caja fuerte)"
    ),
    "illness": re.compile(
        r"\b(intoxic|diarrea|enferm|vomit|fiebre|gastro|estomago|amebi|salmonel|infeccion"
        r"|medico|hospital|clinica|virus|covid|bacteria)"
    ),
    "bad_faith": re.compile(
        r"\b(estaf|engan|timo\b|timad|fraud|menti|mienten|ladron|abus"
        r"|cobr[a-z]* (de mas|indebid)|publicidad enganosa|robo\b)"
    ),
    "returning_guest": re.compile(
        r"\b(repetimos|repito|repetir|volvimos|regresamos|segunda vez|tercera vez|cuarta vez"
        r"|quinta vez|otra vez|de nuevo|ya habiamos|ya habia|nuevamente|como siempre"
        r"|habituales|\d+ ?(a|era|ra|va)? ?vez|\d+ veces)"
    ),
    "with_children": re.compile(
        r"\b(nin[oa]s?|hij[oa]s?|bebes?|peques|pequen[oa]s|kids|nietos?|familia con)\b"
    ),
    "recommends": re.compile(r"\b(recomend|recomiend)"),
    "nights": re.compile(rf"\b(\d+|{_NUMBERS}) (noches|dias)\b|\bsemana\b"),
    "incident": re.compile(
        r"\b(esper|cola|colas|cambi[a-z]* de (habitacion|cuarto)|cobr|reserv|nadie"
        r"|ninguna solucion|no nos (solucionaron|resolvieron))"
    ),
}
ALERT_FIELDS = ("says_no_return", "legal_action", "theft", "illness", "bad_faith")


def has_cue(field: str, text: str) -> bool:
    return CUES[field].search(fold(text)) is not None


def has_any_alert_cue(text: str) -> bool:
    folded = fold(text)
    return any(CUES[f].search(folded) for f in ALERT_FIELDS)
