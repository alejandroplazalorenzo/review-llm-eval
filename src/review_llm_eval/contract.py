"""Output contract: taxonomy, key names and the JSON Schema sent in Ollama's ``format``.

The shape follows the production contract (written from scratch for hotels):

- **Short keys and short enum codes**, expanded to long names in Python
  (``postprocess.expand``). Generation is the bottleneck, so every output token counts
  (experiment E2 measures the saving).
- **Every key is required.** A field that may be empty is nullable *by type*
  (``["integer", "null"]``) or an empty list, never optional: under constrained
  decoding an optional key can simply stop being emitted (experiment E3).
- **Opinions are a list** of ``{topic, polarity, literal quote}``, one per topic the
  guest actually talks about, instead of a fixed object with every topic.
- **Staff names** are a list, checked later against the review text.
- **Alerts are booleans that need an explicit statement**; the incident field is an
  enum whose default value is "none".
- The language is asked (and stored in the raw output only): the stored language comes
  from the deterministic detector (layer 1).

``Variant`` describes the alternative contracts used by the experiments; the pipeline
always uses ``BASE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

JsonObject = dict[str, Any]

CONTRACT_VERSION = "hotel-contract-v1"

# code -> (long name stored in the database, description shown to the model)
TOPICS: dict[str, tuple[str, str]] = {
    "hab": ("room", "la habitación: cama, baño, aire acondicionado, averías, vistas"),
    "lim": ("cleanliness", "limpieza de la habitación o de las zonas comunes"),
    "buf": ("food", "comida: buffet y restaurantes, variedad, calidad, temperatura"),
    "beb": ("drinks", "bebidas y bares, calidad del alcohol"),
    "pis": ("pools", "piscinas y hamacas"),
    "pla": ("beach", "playa, mar, arena, algas"),
    "atn": ("staff_service", "trato del personal: amabilidad, atención, rapidez"),
    "ani": ("entertainment", "animación, espectáculos, actividades, discoteca"),
    "val": ("value", "precio, relación calidad-precio, cargos extra, propinas"),
    "rcp": ("front_desk", "recepción: llegada, salida, reservas, cambios, quejas"),
    "rui": ("noise", "ruido o descanso"),
    "seg": ("safety_health", "seguridad, robos, salud, higiene, medidas sanitarias"),
    "zon": ("grounds_location", "jardines e instalaciones generales, ubicación, traslados"),
}
POLARITIES: dict[str, str] = {"pos": "positive", "neg": "negative", "neu": "neutral"}
INCIDENTS: dict[str, tuple[str | None, str]] = {
    "wait": ("long_wait", "esperó demasiado: habitación sin preparar, colas, llegada lenta"),
    "move": ("room_change", "tuvo que cambiar de habitación o no le dieron la reservada"),
    "charge": ("unexpected_charge", "le cobraron algo inesperado o que no correspondía"),
    "book": ("booking_problem", "no consiguió reservar restaurantes o actividades"),
    "ignored": ("complaint_ignored", "avisó de un problema y nadie lo arregló"),
    "none": (None, "ninguno de los anteriores"),
}
LANGUAGES: dict[str, str] = {
    "es": "spanish",
    "en": "english",
    "pt": "portuguese",
    "fr": "french",
    "it": "italian",
    "de": "german",
    "xx": "other",
}

# Top-level fields in generation order: evidence (opinions, names) before the flags,
# the summary last. (long name, short key)
FIELDS: tuple[tuple[str, str], ...] = (
    ("opinions", "ops"),
    ("staff", "emp"),
    ("incident", "inc"),
    ("says_no_return", "nov"),
    ("legal_action", "leg"),
    ("theft", "rob"),
    ("illness", "enf"),
    ("bad_faith", "fra"),
    ("returning_guest", "ret"),
    ("with_children", "kid"),
    ("recommends", "recom"),
    ("nights", "noc"),
    ("language", "idi"),
    ("summary", "rsm"),
)
OPINION_FIELDS: tuple[tuple[str, str], ...] = (
    ("topic", "t"),
    ("polarity", "p"),
    ("quote", "lit"),
)
ALERTS: tuple[str, ...] = ("says_no_return", "legal_action", "theft", "illness", "bad_faith")
QUALIFIERS: tuple[str, ...] = ("returning_guest", "with_children", "recommends")
BOOLEAN_FIELDS: tuple[str, ...] = ALERTS + QUALIFIERS

KeyStyle = Literal["short", "long"]
SummaryStyle = Literal["words", "numeric"]
Order = Literal["review_first", "instructions_first"]
IncidentStyle = Literal["default_none", "no_default", "phrases", "list", "free_text"]


@dataclass(frozen=True, slots=True)
class Variant:
    """One version of the contract + prompt. ``BASE`` is the production shape."""

    name: str = "base"
    keys: KeyStyle = "short"
    summary: SummaryStyle = "words"
    order: Order = "review_first"
    incident: IncidentStyle = "default_none"
    incident_required: bool = True
    max_lengths: tuple[int, int] | None = None  # (quote, summary) maxLength in characters

    def key(self, long_name: str) -> str:
        """Key used in the model output for a top-level or opinion field."""
        if self.keys == "long":
            return long_name
        return _SHORT_KEYS[long_name]

    def topic_codes(self) -> list[str]:
        return [TOPICS[c][0] for c in TOPICS] if self.keys == "long" else list(TOPICS)

    def polarity_codes(self) -> list[str]:
        return list(POLARITIES.values()) if self.keys == "long" else list(POLARITIES)

    def incident_codes(self, with_none: bool = True) -> list[str]:
        codes = [c for c in INCIDENTS if with_none or c != "none"]
        if self.keys == "long":
            return [INCIDENTS[c][0] or "none" for c in codes]
        return codes

    def language_codes(self) -> list[str]:
        return list(LANGUAGES.values()) if self.keys == "long" else list(LANGUAGES)


_SHORT_KEYS: dict[str, str] = dict(FIELDS) | dict(OPINION_FIELDS)

BASE = Variant()


def build_schema(variant: Variant = BASE) -> JsonObject:
    """JSON Schema for ``variant``: sent to Ollama as ``format`` and used to re-validate
    every answer (the grammar does not stop a generation that runs out of tokens)."""
    k = variant.key
    quote: JsonObject = {"type": "string"}
    summary: JsonObject = {"type": "string"}
    if variant.max_lengths is not None:
        quote["maxLength"], summary["maxLength"] = variant.max_lengths
    opinion = {
        "type": "object",
        "properties": {
            k("topic"): {"type": "string", "enum": variant.topic_codes()},
            k("polarity"): {"type": "string", "enum": variant.polarity_codes()},
            k("quote"): quote,
        },
        "required": [k("topic"), k("polarity"), k("quote")],
    }
    incident: JsonObject
    if variant.incident == "list":
        incident = {
            "type": "array",
            "items": {"type": "string", "enum": variant.incident_codes(with_none=False)},
        }
    elif variant.incident == "free_text":
        # The shape of the first production contract: free text, nullable, and (in
        # experiment E3) left out of ``required``.
        incident = {"type": ["string", "null"]}
    else:
        incident = {"type": "string", "enum": variant.incident_codes()}

    properties: JsonObject = {
        k("opinions"): {"type": "array", "items": opinion},
        k("staff"): {"type": "array", "items": {"type": "string"}},
        k("incident"): incident,
        **{k(name): {"type": "boolean"} for name in BOOLEAN_FIELDS},
        k("nights"): {"type": ["integer", "null"]},
        k("language"): {"type": "string", "enum": variant.language_codes()},
        k("summary"): summary,
    }
    # Re-order to the declared generation order (booleans were inserted as a block).
    ordered = {k(name): properties[k(name)] for name, _ in FIELDS}
    required = [key for key in ordered if variant.incident_required or key != k("incident")]
    return {"type": "object", "properties": ordered, "required": required}
