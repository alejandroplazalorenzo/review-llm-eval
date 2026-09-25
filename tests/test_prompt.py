from __future__ import annotations

import re
from dataclasses import replace

import pytest

from review_llm_eval.contract import BASE, Variant
from review_llm_eval.data import Review
from review_llm_eval.prompt import build_prompt, summary_length


def test_review_goes_first_then_instructions(reviews: list[Review]) -> None:
    prompt = build_prompt(reviews[0])
    assert prompt.index(reviews[0].text) < prompt.index("Devuelve solo el JSON")
    assert prompt.startswith("Reseña de un hotel")


def test_rating_and_title_are_in_the_prompt(reviews: list[Review]) -> None:
    review = replace(reviews[0], rating=2)
    prompt = build_prompt(review)
    assert "Puntuación del huésped: 2 de 5." in prompt
    assert review.title in prompt
    assert "Puntuación del huésped: ? de 5." in build_prompt(replace(review, rating=None))


def test_hotel_name_is_not_sent(reviews: list[Review]) -> None:
    review = replace(reviews[0], hotel="Hotel Very Unique Name")
    assert "Very Unique Name" not in build_prompt(review)


def test_input_is_truncated(reviews: list[Review]) -> None:
    prompt = build_prompt(replace(reviews[0], text="x" * 100_000))
    assert len(prompt) < 10_000


@pytest.mark.parametrize("rating", [None, 1, 2, 3, 4, 5])
def test_summary_length_has_no_number_of_words(reviews: list[Review], rating: int | None) -> None:
    """Decision test: a numeric limit gets echoed inside the summary and can make the
    model run to num_predict (experiment E5). The length is described in words."""
    prompt = build_prompt(replace(reviews[0], rating=rating))
    assert not re.search(r"\d+\s+palabras", prompt)
    summary_line = prompt.splitlines()[-1]
    assert summary_line.startswith('"rsm"')
    assert "palabra" not in summary_line
    assert not re.search(r"\d", summary_line)


def test_summary_is_longer_for_low_ratings() -> None:
    assert summary_length(1) != summary_length(5)
    assert summary_length(3) == summary_length(1)
    assert summary_length(None) == summary_length(5)


def test_numeric_variant_asks_for_a_word_count(reviews: list[Review]) -> None:
    prompt = build_prompt(reviews[0], Variant(summary="numeric"))
    assert re.search(r"\d+ palabras como máximo", prompt)


def test_instructions_first_variant_puts_the_review_last(reviews: list[Review]) -> None:
    prompt = build_prompt(reviews[0], Variant(order="instructions_first"))
    assert prompt.startswith("Devuelve solo el JSON")
    assert prompt.rstrip().endswith('"""')


def test_every_key_of_the_contract_is_explained() -> None:
    from review_llm_eval.contract import FIELDS, OPINION_FIELDS

    prompt = build_prompt(Review(1, 5, "t", "texto", "H", "May 2021", False), BASE)
    for _, short in (*FIELDS, *OPINION_FIELDS):
        assert f'"{short}"' in prompt


def test_alerts_require_an_explicit_statement() -> None:
    prompt = build_prompt(Review(1, 5, "t", "texto", "H", "May 2021", False))
    assert "true solo cuando el texto lo dice con palabras" in prompt
    assert "En caso de duda, false" in prompt
