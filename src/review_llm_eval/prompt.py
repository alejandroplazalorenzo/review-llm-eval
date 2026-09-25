"""Prompt construction: one single prompt for ``/api/generate``, review first.

Decisions carried over from production (the wording is written from scratch):

- The review goes first and the instructions after it, in one prompt (no system
  prompt, no chat turns).
- The star rating is part of the review block: it is context the guest gave, and it
  sets how long the summary may be.
- The summary length is described in words and depends on the rating. A numeric limit
  ("at most N words") is what experiment E5 tests against it.
- Alerts are phrased as "only when the text says it"; the incident enum defaults to
  "none".

The hotel name is not given to the model: it is metadata, stored with the review.
"""

from __future__ import annotations

from review_llm_eval.config import MAX_INPUT_CHARS
from review_llm_eval.contract import BASE, INCIDENTS, TOPICS, Variant
from review_llm_eval.data import Review

PROMPT_VERSION = "prompt-v1"
"""Bump on any change to the prompt or the contract: rows enriched with an older version
are only redone with ``enrich --reprocess``."""

MAX_TITLE_CHARS = 300

NUMERIC_LIMITS = {"long": 30, "short": 15}
"""Word limits used only by the E5 variant ("at most N words")."""


def rating_text(rating: int | None) -> str:
    return "?" if rating is None else str(rating)


def summary_length(rating: int | None, style: str = "words") -> str:
    """How long the summary may be. Low ratings get more room: in this dataset 1-2 star
    reviews average 594 characters against 415 for 4-5 stars (median 716 vs 333), and a
    complaint is what someone will actually read."""
    long_form = rating is not None and rating <= 3
    if style == "numeric":
        limit = NUMERIC_LIMITS["long" if long_form else "short"]
        return f"un resumen de {limit} palabras como máximo"
    return "un par de frases" if long_form else "una frase breve"


def review_block(review: Review) -> str:
    title = review.title[:MAX_TITLE_CHARS] or "(sin título)"
    return (
        "Reseña de un hotel todo incluido de Punta Cana. "
        f"Puntuación del huésped: {rating_text(review.rating)} de 5.\n"
        f"Título: {title}\n"
        f'Texto:\n"""{review.text[:MAX_INPUT_CHARS]}"""'
    )


def _codes(variant: Variant, pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"      {code} = {desc}" for code, desc in pairs)


def _incident_block(variant: Variant) -> str:
    k = variant.key
    codes = variant.incident_codes()
    described = [(codes[i], INCIDENTS[c][1]) for i, c in enumerate(INCIDENTS)]
    if variant.incident == "default_none":
        return (
            f'"{k("incident")}": el contratiempo concreto que el huésped cuenta que tuvo '
            "que resolver durante la estancia. Se queda en "
            f'"{codes[-1]}" salvo que el texto lo cuente claramente:\n' + _codes(variant, described)
        )
    if variant.incident == "no_default":
        return (
            f'"{k("incident")}": el contratiempo concreto que el huésped tuvo que '
            "resolver durante la estancia:\n" + _codes(variant, described)
        )
    if variant.incident == "phrases":
        phrases = {
            "wait": "«tuvimos que esperar horas», «la habitación no estaba lista»",
            "move": "«nos cambiaron de habitación», «no era la que reservamos»",
            "charge": "«nos cobraron», «cargo extra», «nos quisieron cobrar»",
            "book": "«imposible reservar», «no había mesa en los restaurantes»",
            "ignored": "«lo reportamos y nadie vino», «nadie nos dio solución»",
            "none": "cualquier otro caso",
        }
        return (
            f'"{k("incident")}": el contratiempo que el huésped cuenta con frases como '
            "estas; si no hay ninguna parecida, "
            f'"{codes[-1]}":\n'
            + _codes(variant, [(codes[i], phrases[c]) for i, c in enumerate(INCIDENTS)])
        )
    if variant.incident == "list":
        return (
            f'"{k("incident")}": lista de los contratiempos concretos que el huésped '
            "cuenta que tuvo que resolver (puede estar vacía):\n" + _codes(variant, described[:-1])
        )
    # free_text (experiment E3, the shape of the first production contract)
    return (
        f'"{k("incident")}": si el huésped cuenta un contratiempo concreto que tuvo que '
        "resolver (esperas, cambios de habitación, cobros, reservas, quejas sin "
        "solución), resúmelo en una frase; si no, null."
    )


def instructions(variant: Variant, rating: int | None) -> str:
    k = variant.key
    topics = variant.topic_codes()
    polarity = variant.polarity_codes()
    languages = variant.language_codes()
    topic_lines = _codes(variant, [(topics[i], TOPICS[c][1]) for i, c in enumerate(TOPICS)])
    lang_desc = ", ".join(languages[:-1]) + f" u {languages[-1]} (otro)"
    return f"""Devuelve solo el JSON pedido. Las claves y los códigos van tal cual.

"{k("opinions")}": un objeto por cada tema sobre el que el huésped da una opinión concreta. No repitas un tema con la misma polaridad.
  "{k("topic")}":
{topic_lines}
  "{k("polarity")}": {polarity[0]}, {polarity[1]} o {polarity[2]}
  "{k("quote")}": un trozo breve de la reseña, copiado letra por letra, que muestre esa opinión.
  Si no opina de nada concreto: [].

"{k("staff")}": nombres propios de trabajadores del hotel escritos en la reseña (camareros, animadores, mayordomos, recepcionistas, cocineros...). Deja fuera al autor y a quienes viajaban con él, los puestos sin nombre ("el mayordomo") y los nombres de hoteles, cadenas o lugares. Si no hay ninguno: [].

{_incident_block(variant)}

Alertas: true solo cuando el texto lo dice con palabras; deducirlo del tono no vale. En caso de duda, false.
"{k("says_no_return")}": escribe que no volverá o que desaconseja el hotel ("no vuelvo", "nunca más", "no lo recomiendo").
"{k("legal_action")}": habla de denuncia, abogado, organismo de consumo, reclamación formal o de reclamar al banco o a la tarjeta.
"{k("theft")}": cuenta que les robaron o que les desapareció algo.
"{k("illness")}": cuenta que alguien enfermó durante la estancia (intoxicación, diarrea, fiebre).
"{k("bad_faith")}": acusa al hotel de engañar a propósito: estafa, timo, mentiras, publicidad engañosa, cobros que sabían indebidos. Quejarse del servicio no es esto.

Sobre el huésped:
"{k("returning_guest")}": true si dice que ya se había alojado antes en este hotel.
"{k("with_children")}": true si viajaba con niños o bebés.
"{k("recommends")}": true solo si recomienda el hotel de forma expresa ("lo recomiendo", "recomendable"). Hablar bien no basta.
"{k("nights")}": cuántas noches duró la estancia si el texto lo dice (una semana son siete); si no lo dice, null.
"{k("language")}": idioma en que está escrita la reseña: {lang_desc}.

"{k("summary")}": {summary_length(rating, variant.summary)} en español contando qué le pasó al huésped (hechos, no su estado de ánimo). Escribe solo el resumen."""


def build_prompt(review: Review, variant: Variant = BASE) -> str:
    """The single prompt sent to ``/api/generate`` for ``review``."""
    block = review_block(review)
    body = instructions(variant, review.rating)
    if variant.order == "instructions_first":
        return f"{body}\n\n{block}"
    return f"{block}\n\n{body}"
