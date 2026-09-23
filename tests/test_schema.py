from __future__ import annotations

import json

import pytest

from helpers import make_output
from review_llm_eval.config import ASPECTS
from review_llm_eval.schema import (
    COMPLAINT_MAX_CHARS,
    build_schema,
    parse_and_validate,
    validate_output,
)


@pytest.fixture
def schema() -> dict:
    return build_schema()


def test_valid_output_passes(schema: dict) -> None:
    out = make_output("mixed", complaint="Dirty pool", would_return=False, staff="positive")
    assert validate_output(out, schema) == []


def test_all_nulls_are_valid(schema: dict) -> None:
    assert validate_output(make_output("neutral"), schema) == []


def test_every_field_is_required(schema: dict) -> None:
    assert schema["required"] == ["aspects", "complaint", "would_return", "sentiment"]
    assert schema["properties"]["aspects"]["required"] == list(ASPECTS)
    out = make_output()
    del out["would_return"]
    errors = validate_output(out, schema)
    assert any("'would_return' is a required property" in e for e in errors)


def test_missing_aspect_key_fails(schema: dict) -> None:
    out = make_output()
    del out["aspects"]["noise"]
    assert validate_output(out, schema)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o.update(sentiment="very positive"),
        lambda o: o["aspects"].update(staff="good"),
        lambda o: o.update(would_return="yes"),
        lambda o: o.update(complaint="x" * (COMPLAINT_MAX_CHARS + 1)),
        lambda o: o.update(extra_field=1),
        lambda o: o["aspects"].update(spa="positive"),
    ],
    ids=["bad-sentiment", "bad-aspect-value", "bad-bool", "long-complaint", "extra", "new-aspect"],
)
def test_invalid_outputs_fail(schema: dict, mutate) -> None:
    out = make_output()
    mutate(out)
    assert validate_output(out, schema)


def test_optional_variant_accepts_missing_flags() -> None:
    schema = build_schema("optional")
    out = make_output()
    del out["complaint"], out["would_return"]
    assert validate_output(out, schema) == []
    assert validate_output(out, build_schema("required"))


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        build_schema("loose")  # type: ignore[arg-type]


def test_parse_and_validate_ok(schema: dict) -> None:
    out = make_output("negative", complaint="Cold food", food="negative")
    obj, err = parse_and_validate(json.dumps(out), schema)
    assert err is None and obj == out


def test_parse_and_validate_rejects_truncated_json(schema: dict) -> None:
    raw = json.dumps(make_output())[:40]  # e.g. generation stopped at the token limit
    obj, err = parse_and_validate(raw, schema)
    assert obj is None and err is not None and err.startswith("invalid JSON")


def test_parse_and_validate_reports_schema_error(schema: dict) -> None:
    obj, err = parse_and_validate('{"sentiment": "positive"}', schema)
    assert obj is None and err is not None and "required property" in err
