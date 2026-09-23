"""Dataset loading, de-duplication and deterministic stratified sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE_TRUNCATION_CHARS = 780
"""The public dataset cuts long reviews at ~790 characters (scraped previews).
Reviews at or above this length are flagged ``truncated_in_source``."""


@dataclass(frozen=True, slots=True)
class Review:
    review_id: int  # row index in the Hugging Face parquet export
    rating: int
    title: str
    text: str
    hotel: str
    wrote: str
    truncated_in_source: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Review:
        return cls(
            review_id=int(row["review_id"]),
            rating=int(row["rating"]),
            title=str(row["title"]),
            text=str(row["text"]),
            hotel=str(row["hotel"]),
            wrote=str(row["wrote"]),
            truncated_in_source=bool(row["truncated_in_source"]),
        )


def rows_to_reviews(rows: Iterable[dict[str, Any]]) -> list[Review]:
    """Convert raw dataset rows (in file order) into reviews.

    Drops rows with an empty text or an out-of-range rating and keeps only the first
    occurrence of each duplicated review text (the public dump contains ~1.5k repeats).
    """
    seen: set[str] = set()
    reviews: list[Review] = []
    for index, row in enumerate(rows):
        text = (row.get("review_text") or "").strip()
        rating = row.get("rating")
        if not text or rating not in (1, 2, 3, 4, 5):
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        reviews.append(
            Review(
                review_id=index,
                rating=int(rating),
                title=(row.get("title") or "").strip(),
                text=text,
                hotel=(row.get("hotel_name") or "").strip(),
                wrote=(row.get("wrote") or "").strip(),
                truncated_in_source=len(text) >= SOURCE_TRUNCATION_CHARS,
            )
        )
    return reviews


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read the raw parquet file into a list of dicts (pyarrow imported lazily)."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def stratified_sample(reviews: Sequence[Review], per_rating: int, seed: int) -> list[Review]:
    """Draw ``per_rating`` reviews from each star rating (equal allocation).

    Deterministic for a given input order and seed: every stratum is sorted by
    ``review_id`` before sampling, and a single ``random.Random(seed)`` is consumed
    in rating order. The result is sorted by (rating, review_id).
    """
    by_rating: dict[int, list[Review]] = defaultdict(list)
    for review in reviews:
        by_rating[review.rating].append(review)
    rng = random.Random(seed)
    sample: list[Review] = []
    for rating in sorted(by_rating):
        stratum = sorted(by_rating[rating], key=lambda r: r.review_id)
        if len(stratum) < per_rating:
            raise ValueError(f"rating {rating} has {len(stratum)} reviews, fewer than {per_rating}")
        sample.extend(rng.sample(stratum, per_rating))
    return sorted(sample, key=lambda r: (r.rating, r.review_id))


def gold_queue(sample: Sequence[Review], total: int, seed: int) -> list[Review]:
    """Deterministic subset of the sample, balanced by rating, for human labelling.

    The order is shuffled (seeded) so that a partially labelled gold file still
    covers every rating.
    """
    ratings = sorted({r.rating for r in sample})
    per_rating = total // len(ratings)
    rng = random.Random(seed)
    chosen: list[Review] = []
    for rating in ratings:
        stratum = sorted((r for r in sample if r.rating == rating), key=lambda r: r.review_id)
        chosen.extend(rng.sample(stratum, min(per_rating, len(stratum))))
    rng.shuffle(chosen)
    return chosen
