from __future__ import annotations

from dataclasses import replace

from review_llm_eval.config import ASPECTS
from review_llm_eval.data import Review
from review_llm_eval.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    build_retry_messages,
    format_review,
)


def test_messages_are_system_then_user(reviews: list[Review]) -> None:
    messages = build_messages(reviews[0])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert reviews[0].text in messages[1]["content"]
    assert reviews[0].title in messages[1]["content"]


def test_rating_and_hotel_never_reach_the_model(reviews: list[Review]) -> None:
    review = replace(reviews[0], rating=1, hotel="Hotel Very Unique Name")
    content = " ".join(m["content"] for m in build_messages(review))
    assert "Hotel Very Unique Name" not in content
    assert "rating" not in content.lower() and "stars" not in content.lower()


def test_every_aspect_is_defined_in_the_prompt() -> None:
    for aspect in ASPECTS:
        assert f"- {aspect}:" in SYSTEM_PROMPT


def test_empty_title_is_explicit(reviews: list[Review]) -> None:
    assert format_review(replace(reviews[0], title="")).startswith("Title: (no title)")


def test_retry_messages_keep_original_and_add_feedback(reviews: list[Review]) -> None:
    base = build_messages(reviews[0])
    retry = build_retry_messages(base, '{"bad": 1}', "sentiment: required")
    assert retry[:2] == base
    assert retry[2] == {"role": "assistant", "content": '{"bad": 1}'}
    assert "sentiment: required" in retry[3]["content"]
    assert len(base) == 2  # the original list is not mutated
