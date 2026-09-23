from __future__ import annotations

from collections import Counter

import pytest

from review_llm_eval.data import (
    SOURCE_TRUNCATION_CHARS,
    Review,
    gold_queue,
    rows_to_reviews,
    stratified_sample,
)
from review_llm_eval.run import reviews_for_limit


def synthetic_reviews(per_rating: int = 30) -> list[Review]:
    rows = [
        {
            "hotel_name": "H",
            "wrote": "May 2021",
            "rating": rating,
            "title": f"t{rating}-{i}",
            "review_text": f"texto {rating} {i}",
        }
        for i in range(per_rating)
        for rating in (1, 2, 3, 4, 5)
    ]
    return rows_to_reviews(rows)


def test_rows_to_reviews_cleans_and_deduplicates() -> None:
    rows = [
        {"rating": 5, "title": "a", "review_text": "Muy bien", "hotel_name": "H", "wrote": "x"},
        {"rating": 4, "title": "b", "review_text": "  muy   BIEN ", "hotel_name": "H"},  # dup
        {"rating": 1, "title": "c", "review_text": "", "hotel_name": "H"},  # empty
        {"rating": 7, "title": "d", "review_text": "raro", "hotel_name": "H"},  # bad rating
        {"rating": 2, "title": None, "review_text": "x" * SOURCE_TRUNCATION_CHARS},
    ]
    reviews = rows_to_reviews(rows)
    assert [r.review_id for r in reviews] == [0, 4]  # ids are original row indices
    assert reviews[1].title == "" and reviews[1].truncated_in_source
    assert not reviews[0].truncated_in_source


def test_review_round_trip(reviews: list[Review]) -> None:
    assert [Review.from_dict(r.to_dict()) for r in reviews] == reviews


def test_stratified_sample_is_deterministic_and_balanced() -> None:
    pool = synthetic_reviews()
    first = stratified_sample(pool, per_rating=7, seed=42)
    second = stratified_sample(list(reversed(pool)), per_rating=7, seed=42)
    assert first == second  # input order does not matter
    assert Counter(r.rating for r in first) == {1: 7, 2: 7, 3: 7, 4: 7, 5: 7}
    assert len({r.review_id for r in first}) == 35


def test_stratified_sample_changes_with_seed() -> None:
    pool = synthetic_reviews()
    assert stratified_sample(pool, 7, seed=1) != stratified_sample(pool, 7, seed=2)


def test_stratified_sample_rejects_small_strata() -> None:
    with pytest.raises(ValueError):
        stratified_sample(synthetic_reviews(per_rating=3), per_rating=5, seed=0)


def test_gold_queue_is_deterministic_balanced_and_shuffled() -> None:
    sample = stratified_sample(synthetic_reviews(), per_rating=10, seed=42)
    queue = gold_queue(sample, total=20, seed=42)
    assert queue == gold_queue(sample, total=20, seed=42)
    assert Counter(r.rating for r in queue) == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
    assert [r.rating for r in queue] != sorted(r.rating for r in queue)


def test_reviews_for_limit_round_robins_over_ratings(reviews: list[Review]) -> None:
    picked = reviews_for_limit(reviews, 7)
    assert [r.rating for r in picked] == [1, 2, 3, 4, 5, 1, 2]
    assert len(reviews_for_limit(reviews, 100)) == len(reviews)
