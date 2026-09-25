"""Dataset rows -> ``Review`` objects: normalisation, de-duplication and sampling.

Normalisation follows the rules of the production pipeline, adapted to this dataset:

- a bad field degrades to ``None`` with a warning; a review is never dropped for it
  (the only rows left out are empty texts and exact repeats of an earlier text);
- the date is metadata, not something to ask the model: ``wrote`` ("August 2021") gives
  a month, so every date here has *month* precision;
- the content hash covers everything the model sees (rating, title, text), so a change
  in any of them makes the stored enrichment stale.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SOURCE_TRUNCATION_CHARS = 780
"""The public dataset cuts long reviews at ~790 characters (scraped previews).
Reviews at or above this length are flagged ``truncated_in_source``."""

MONTH_PRECISION = "month"


@dataclass(frozen=True, slots=True)
class Review:
    review_id: int  # row index in the Hugging Face parquet export
    rating: int | None
    title: str
    text: str
    hotel: str
    wrote: str
    truncated_in_source: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Review:
        rating = row.get("rating")
        return cls(
            review_id=int(row["review_id"]),
            rating=None if rating is None else int(rating),
            title=str(row.get("title") or ""),
            text=str(row["text"]),
            hotel=str(row.get("hotel") or ""),
            wrote=str(row.get("wrote") or ""),
            truncated_in_source=bool(row.get("truncated_in_source", False)),
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self.rating, self.title, self.text)

    @property
    def full_text(self) -> str:
        """Title and body, the text quotes and staff names are checked against."""
        return f"{self.title}\n{self.text}" if self.title else self.text


def content_hash(rating: int | None, title: str, text: str) -> str:
    payload = f"{rating}\x1f{title}\x1f{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def month_from_wrote(wrote: str | None) -> tuple[str | None, str | None, str | None]:
    """``"August 2021"`` -> ``("2021-08", "month", None)``.

    Returns ``(month, precision, warning)``; a missing or unreadable value gives
    ``(None, None, warning-or-None)``.
    """
    if not wrote or not wrote.strip():
        return None, None, None
    parts = wrote.split()
    month = _MONTHS.get(parts[0].lower()) if len(parts) == 2 else None
    if month is None or not parts[1].isdigit() or len(parts[1]) != 4:
        return None, None, f"unreadable 'wrote' value: {wrote!r}"
    return f"{parts[1]}-{month:02d}", MONTH_PRECISION, None


_MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
"""English month names, parsed explicitly so the result never depends on the locale."""


def normalize_rating(value: Any) -> tuple[int | None, str | None]:
    """1-5 or ``None`` with a warning (a bad rating never drops the review)."""
    if value is None:
        return None, None
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return None, f"unreadable rating: {value!r}"
    if not 1 <= rating <= 5:
        return None, f"rating out of range: {value!r}"
    return rating, None


def rows_to_reviews(rows: Iterable[dict[str, Any]]) -> tuple[list[Review], list[str]]:
    """Convert raw dataset rows (in file order) into reviews.

    Keeps only the first occurrence of each review text (the public dump contains ~1.5k
    exact repeats) and skips empty texts. Returns ``(reviews, warnings)``.
    """
    seen: set[str] = set()
    reviews: list[Review] = []
    warnings: list[str] = []
    empty = repeats = 0
    for index, row in enumerate(rows):
        text = (row.get("review_text") or "").strip()
        if not text:
            empty += 1
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        rating, warning = normalize_rating(row.get("rating"))
        if warning:
            warnings.append(f"row {index}: {warning}")
        reviews.append(
            Review(
                review_id=index,
                rating=rating,
                title=(row.get("title") or "").strip(),
                text=text,
                hotel=(row.get("hotel_name") or "").strip(),
                wrote=(row.get("wrote") or "").strip(),
                truncated_in_source=len(text) >= SOURCE_TRUNCATION_CHARS,
            )
        )
    if empty:
        warnings.append(f"{empty} row(s) with an empty text skipped")
    if repeats:
        warnings.append(f"{repeats} row(s) repeating an earlier text skipped")
    return reviews, warnings


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read the raw parquet file into a list of dicts (pyarrow imported lazily)."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def stratified_sample(reviews: Sequence[Review], per_rating: int, seed: int) -> list[Review]:
    """Draw ``per_rating`` reviews from each star rating (equal allocation).

    Deterministic for a given input and seed: every stratum is sorted by ``review_id``
    before sampling and one ``random.Random(seed)`` is consumed in rating order. The
    result is sorted by (rating, review_id). Reviews without a rating are not sampled.
    """
    by_rating: dict[int, list[Review]] = defaultdict(list)
    for review in reviews:
        if review.rating is not None:
            by_rating[review.rating].append(review)
    rng = random.Random(seed)
    sample: list[Review] = []
    for rating in sorted(by_rating):
        stratum = sorted(by_rating[rating], key=lambda r: r.review_id)
        if len(stratum) < per_rating:
            raise ValueError(f"rating {rating} has {len(stratum)} reviews, fewer than {per_rating}")
        sample.extend(rng.sample(stratum, per_rating))
    return sorted(sample, key=lambda r: (r.rating or 0, r.review_id))


def round_robin(reviews: Sequence[Review], limit: int | None = None) -> list[Review]:
    """Reviews reordered round-robin over star ratings, optionally cut to ``limit``, so a
    small experimental subset still covers every rating."""
    by_rating: dict[int, list[Review]] = {}
    for review in sorted(reviews, key=lambda r: r.review_id):
        by_rating.setdefault(review.rating or 0, []).append(review)
    queues = [by_rating[k] for k in sorted(by_rating)]
    ordered: list[Review] = []
    total = len(reviews) if limit is None else min(limit, len(reviews))
    index = 0
    while len(ordered) < total:
        for queue in queues:
            if index < len(queue) and len(ordered) < total:
                ordered.append(queue[index])
        index += 1
    return ordered


def gold_queue(sample: Sequence[Review], total: int, seed: int) -> list[Review]:
    """Deterministic subset of the sample, balanced by rating, for human labelling.

    The order is shuffled (seeded) so that a partially labelled gold file still covers
    every rating.
    """
    ratings = sorted({r.rating for r in sample if r.rating is not None})
    per_rating = total // len(ratings)
    rng = random.Random(seed)
    chosen: list[Review] = []
    for rating in ratings:
        stratum = sorted((r for r in sample if r.rating == rating), key=lambda r: r.review_id)
        chosen.extend(rng.sample(stratum, min(per_rating, len(stratum))))
    rng.shuffle(chosen)
    return chosen
