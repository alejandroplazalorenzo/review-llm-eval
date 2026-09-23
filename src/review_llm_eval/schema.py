"""Output contract: the JSON Schema sent to the model and used to validate its answer.

The same schema object does two jobs:

1. Ollama turns it into a grammar that constrains decoding (``format`` field).
2. ``jsonschema`` re-validates every answer, because the grammar is not a guarantee
   (generation can stop at the token limit, and not every keyword is enforced).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from jsonschema import Draft202012Validator

from review_llm_eval.config import ASPECT_SENTIMENTS, ASPECTS, OVERALL_SENTIMENTS

SCHEMA_VERSION = "1.0"
COMPLAINT_MAX_CHARS = 120

SchemaVariant = Literal["required", "optional"]

JsonObject = dict[str, Any]


def build_schema(variant: SchemaVariant = "required") -> JsonObject:
    """Return the output schema.

    ``required`` (production): every field is required; "unknown" is expressed as
    ``null`` through nullable types. ``optional``: identical except that
    ``would_return`` and ``complaint`` are left out of ``required``; it only exists
    to observe how optional keys behave under constrained decoding.
    """
    aspect_value = {
        "type": ["string", "null"],
        "enum": [*ASPECT_SENTIMENTS, None],
    }
    # Property order matters: under constrained decoding the model writes fields in
    # this order, so per-aspect evidence comes before the overall verdict.
    properties: JsonObject = {
        "aspects": {
            "type": "object",
            "properties": {aspect: aspect_value for aspect in ASPECTS},
            "required": list(ASPECTS),
            "additionalProperties": False,
        },
        "complaint": {"type": ["string", "null"], "maxLength": COMPLAINT_MAX_CHARS},
        "would_return": {"type": ["boolean", "null"]},
        "sentiment": {"type": "string", "enum": list(OVERALL_SENTIMENTS)},
    }
    required = list(properties)
    if variant == "optional":
        required = [k for k in required if k not in ("complaint", "would_return")]
    elif variant != "required":
        raise ValueError(f"unknown schema variant: {variant}")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def validate_output(obj: Any, schema: JsonObject) -> list[str]:
    """Return a sorted list of human-readable validation errors (empty if valid)."""
    validator = Draft202012Validator(schema)
    errors = []
    for err in validator.iter_errors(obj):
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{location}: {err.message}")
    return sorted(errors)


def parse_and_validate(raw: str, schema: JsonObject) -> tuple[JsonObject | None, str | None]:
    """Parse raw model text and validate it.

    Returns ``(obj, None)`` on success, ``(None, error)`` otherwise.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at char {exc.pos}"
    errors = validate_output(obj, schema)
    if errors:
        return None, "; ".join(errors[:5])
    return obj, None
