"""The output contract: every key required, nullable only by type, short keys."""

from __future__ import annotations

import json

import pytest

from helpers import make_output
from review_llm_eval.contract import (
    BASE,
    FIELDS,
    INCIDENTS,
    TOPICS,
    Variant,
    build_schema,
)
from review_llm_eval.extract import parse_and_validate, validator_for

VALIDATOR = validator_for(build_schema(BASE))


def valid(obj: object) -> bool:
    return parse_and_validate(json.dumps(obj), VALIDATOR)[0] is not None


def test_reference_output_is_valid() -> None:
    assert valid(make_output())


@pytest.mark.parametrize("key", [short for _, short in FIELDS])
def test_no_field_is_optional(key: str) -> None:
    """Decision test. Under constrained decoding an optional key can stop being emitted
    (experiment E3), so every key is required, including the empty-able ones."""
    out = make_output()
    del out[key]
    assert not valid(out)


def test_empty_values_are_expressed_by_type_not_by_absence() -> None:
    assert valid(make_output(ops=[], emp=[], inc="none", noc=None))
    assert not valid(make_output(nov=None))  # alerts are plain booleans, never null


def test_opinion_items_need_topic_polarity_and_quote() -> None:
    assert not valid(make_output(ops=[{"t": "hab", "p": "neg"}]))
    assert not valid(make_output(ops=[{"t": "spa", "p": "neg", "lit": "x"}]))
    assert not valid(make_output(ops=[{"t": "hab", "p": "mixed", "lit": "x"}]))


def test_enums_are_the_short_codes() -> None:
    schema = build_schema(BASE)["properties"]
    assert schema["inc"]["enum"] == list(INCIDENTS)
    assert schema["ops"]["items"]["properties"]["t"]["enum"] == list(TOPICS)
    assert "none" in schema["inc"]["enum"]  # the default value of the enum


def test_keys_follow_the_generation_order() -> None:
    assert list(build_schema(BASE)["properties"]) == [short for _, short in FIELDS]


def test_long_variant_uses_long_names() -> None:
    schema = build_schema(Variant(keys="long"))
    assert list(schema["properties"])[:3] == ["opinions", "staff", "incident"]
    assert "room" in schema["properties"]["opinions"]["items"]["properties"]["topic"]["enum"]
    assert "long_wait" in schema["properties"]["incident"]["enum"]


def test_optional_variant_only_drops_the_incident() -> None:
    schema = build_schema(Variant(incident="free_text", incident_required=False))
    assert "inc" not in schema["required"]
    assert schema["properties"]["inc"] == {"type": ["string", "null"]}
    assert len(schema["required"]) == len(FIELDS) - 1


def test_max_length_variant() -> None:
    schema = build_schema(Variant(max_lengths=(80, 200)))
    assert schema["properties"]["ops"]["items"]["properties"]["lit"]["maxLength"] == 80
    assert schema["properties"]["rsm"]["maxLength"] == 200
    assert "maxLength" not in json.dumps(build_schema(BASE))


def test_incident_list_variant_has_no_none() -> None:
    items = build_schema(Variant(incident="list"))["properties"]["inc"]["items"]
    assert "none" not in items["enum"]
