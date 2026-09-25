from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from review_llm_eval.data import (
    SOURCE_TRUNCATION_CHARS,
    Review,
    content_hash,
    gold_queue,
    month_from_wrote,
    round_robin,
    rows_to_reviews,
    stratified_sample,
)


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
    return rows_to_reviews(rows)[0]


def test_rows_to_reviews_cleans_deduplicates_and_never_drops_for_a_bad_rating() -> None:
    rows = [
        {"rating": 5, "title": "a", "review_text": "Muy bien", "hotel_name": "H", "wrote": "x"},
        {"rating": 4, "title": "b", "review_text": "  muy   BIEN ", "hotel_name": "H"},  # dup
        {"rating": 1, "title": "c", "review_text": "", "hotel_name": "H"},  # empty
        {"rating": 7, "title": "d", "review_text": "raro", "hotel_name": "H"},  # bad rating
        {"rating": 2, "title": None, "review_text": "x" * SOURCE_TRUNCATION_CHARS},
    ]
    reviews, warnings = rows_to_reviews(rows)
    assert [r.review_id for r in reviews] == [0, 3, 4]  # ids are original row indices
    assert reviews[1].rating is None  # kept, rating set to None
    assert reviews[2].title == "" and reviews[2].truncated_in_source
    assert any("out of range" in w for w in warnings)
    assert any("1 row(s) with an empty text" in w for w in warnings)
    assert any("1 row(s) repeating" in w for w in warnings)


def test_month_from_wrote() -> None:
    assert month_from_wrote("August 2021") == ("2021-08", "month", None)
    assert month_from_wrote("") == (None, None, None)
    month, precision, warning = month_from_wrote("Agosto de 2021")
    assert month is None and precision is None and warning


def test_content_hash_covers_rating_title_and_text(reviews: list[Review]) -> None:
    r = reviews[0]
    assert r.content_hash == content_hash(r.rating, r.title, r.text)
    assert replace(r, rating=5).content_hash != r.content_hash
    assert replace(r, title="x").content_hash != r.content_hash
    assert replace(r, hotel="other").content_hash == r.content_hash


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


def test_round_robin_over_ratings(reviews: list[Review]) -> None:
    picked = round_robin(reviews, 7)
    assert [r.rating for r in picked] == [1, 2, 3, 4, 5, 1, 2]
    assert len(round_robin(reviews)) == len(reviews)
