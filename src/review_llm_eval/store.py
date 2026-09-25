"""SQLite store: raw reviews and their enrichment in separate tables.

``review`` holds only source data (never model output). ``review_enrichment`` can be
emptied and rebuilt from ``review`` at any time: layer 1 fills the first columns, the
LLM layer the rest, and ``raw_output`` keeps the model's JSON exactly as it came so a
mapping change can be re-applied without the GPU.

"Pending" is a query, not a flag: a killed run loses at most the uncommitted batch and
the next run continues from there.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_llm_eval.data import Review, month_from_wrote

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review (
    review_id           INTEGER PRIMARY KEY,  -- row index in the Hugging Face parquet
    hotel               TEXT NOT NULL,
    rating              INTEGER CHECK (rating BETWEEN 1 AND 5),  -- NULL if unreadable
    title               TEXT NOT NULL,
    text                TEXT NOT NULL,
    wrote               TEXT,
    month               TEXT,                 -- 'YYYY-MM' derived from wrote
    month_precision     TEXT CHECK (month_precision IN ('month')),
    truncated_in_source INTEGER NOT NULL,
    content_hash        TEXT NOT NULL,
    loaded_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_enrichment (
    review_id         INTEGER PRIMARY KEY REFERENCES review (review_id),
    content_hash      TEXT NOT NULL,  -- hash of the review content this row was computed on
    model_version     TEXT NOT NULL,  -- '<layer1>' or '<layer1>+<model>/<prompt>'
    -- layer 1
    sentiment         TEXT CHECK (sentiment IN ('POS', 'NEU', 'NEG')),
    sentiment_score   REAL,
    emotion           TEXT,
    language          TEXT,
    -- LLM layer
    opinions          TEXT,  -- JSON list of {topic, polarity, quote}
    staff             TEXT,  -- JSON list of normalized names
    incident          TEXT,
    says_no_return    INTEGER,
    legal_action      INTEGER,
    theft             INTEGER,
    illness           INTEGER,
    bad_faith         INTEGER,
    returning_guest   INTEGER,
    with_children     INTEGER,
    recommends        INTEGER,
    nights            INTEGER,
    summary           TEXT,
    raw_output        TEXT,  -- the model's JSON exactly as returned
    llm_attempts      INTEGER,
    enriched_at       TEXT
);

CREATE VIEW IF NOT EXISTS v_staff_mentions AS
SELECT j.value AS staff_name, r.hotel, count(*) AS mentions
FROM review_enrichment e
JOIN review r USING (review_id)
JOIN json_each(e.staff) j
WHERE e.staff IS NOT NULL
GROUP BY j.value, r.hotel;

CREATE VIEW IF NOT EXISTS v_hotel AS
SELECT hotel,
       count(*)                                        AS n_reviews,
       avg(rating)                                     AS mean_rating,
       avg(CASE WHEN rating <= 2 THEN 1.0 ELSE 0 END) AS share_low
FROM review
GROUP BY hotel;
"""

LLM_COLUMNS: tuple[str, ...] = (
    "opinions",
    "staff",
    "incident",
    "says_no_return",
    "legal_action",
    "theft",
    "illness",
    "bad_faith",
    "returning_guest",
    "with_children",
    "recommends",
    "nights",
    "summary",
    "raw_output",
    "llm_attempts",
)
"""Every column the LLM layer writes. Layer 1 empties all of them when it rewrites a row
(a test checks that this list and ``write_llm`` stay in sync)."""


@dataclass(frozen=True, slots=True)
class Scope:
    """Which reviews an ``enrich`` run may touch (``None`` = no restriction)."""

    review_ids: frozenset[int] | None = None
    hotel: str | None = None

    def sql(self, conn: sqlite3.Connection, alias: str = "r") -> tuple[str, list[Any]]:
        """SQL condition (starting with AND) and its parameters. Review ids go through a
        temporary table, so a scope of any size stays under SQLite's variable limit."""
        clauses: list[str] = []
        params: list[Any] = []
        if self.review_ids is not None:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS scope_ids (id INTEGER PRIMARY KEY)")
            conn.execute("DELETE FROM temp.scope_ids")
            conn.executemany(
                "INSERT INTO temp.scope_ids VALUES (?)", [(i,) for i in self.review_ids]
            )
            clauses.append(f"{alias}.review_id IN (SELECT id FROM temp.scope_ids)")
        if self.hotel is not None:
            clauses.append(f"{alias}.hotel = ?")
            params.append(self.hotel)
        return ("".join(f" AND {c}" for c in clauses), params)


ALL = Scope()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    return conn


# --- raw reviews ------------------------------------------------------------------------


def upsert_reviews(
    conn: sqlite3.Connection, reviews: Iterable[Review]
) -> tuple[int, int, int, list[str]]:
    """Insert new reviews and update changed ones. Returns
    ``(inserted, updated, unchanged, warnings)``; a second run over the same data writes
    nothing."""
    existing = dict(conn.execute("SELECT review_id, content_hash FROM review").fetchall())
    inserted = updated = unchanged = 0
    warnings: list[str] = []
    stamp = now()
    for review in reviews:
        month, precision, warning = month_from_wrote(review.wrote)
        if warning:
            warnings.append(f"review {review.review_id}: {warning}")
        values = (
            review.review_id,
            review.hotel,
            review.rating,
            review.title,
            review.text,
            review.wrote or None,
            month,
            precision,
            int(review.truncated_in_source),
            review.content_hash,
            stamp,
        )
        previous = existing.get(review.review_id)
        if previous == review.content_hash:
            unchanged += 1
            continue
        conn.execute(
            """INSERT INTO review (review_id, hotel, rating, title, text, wrote, month,
                                   month_precision, truncated_in_source, content_hash,
                                   loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (review_id) DO UPDATE SET
                   hotel = excluded.hotel, rating = excluded.rating,
                   title = excluded.title, text = excluded.text, wrote = excluded.wrote,
                   month = excluded.month, month_precision = excluded.month_precision,
                   truncated_in_source = excluded.truncated_in_source,
                   content_hash = excluded.content_hash, loaded_at = excluded.loaded_at""",
            values,
        )
        if previous is None:
            inserted += 1
        else:
            updated += 1
    conn.commit()
    return inserted, updated, unchanged, warnings


# --- layer 1 ----------------------------------------------------------------------------


def pending_layer1(
    conn: sqlite3.Connection,
    scope: Scope,
    limit: int,
    reprocess: bool,
    layer1_version: str,
    full_version: str,
) -> list[sqlite3.Row]:
    """Rows without layer 1, whose content changed since, or (with ``reprocess``) whose
    version is neither the current layer-1 nor the current full version."""
    where, params = scope.sql(conn)
    extra = "OR e.model_version NOT IN (?, ?)" if reprocess else ""
    if reprocess:
        params = [*params, layer1_version, full_version]
    return conn.execute(
        f"""SELECT r.review_id, r.title, r.text, r.content_hash
            FROM review r LEFT JOIN review_enrichment e USING (review_id)
            WHERE r.text <> '' {where}
              AND (e.review_id IS NULL OR e.sentiment IS NULL
                   OR e.content_hash <> r.content_hash {extra})
            ORDER BY r.review_id LIMIT ?""",
        [*params, limit],
    ).fetchall()


def write_layer1(conn: sqlite3.Connection, rows: Sequence[tuple[Any, ...]]) -> None:
    """Upsert layer-1 results ``(review_id, content_hash, sentiment, score, emotion,
    language, version)`` and empty every LLM column of those rows."""
    cleared = ", ".join(f"{c} = NULL" for c in LLM_COLUMNS)
    conn.executemany(
        f"""INSERT INTO review_enrichment (review_id, content_hash, sentiment,
                                           sentiment_score, emotion, language,
                                           model_version, enriched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, '{now()}')
            ON CONFLICT (review_id) DO UPDATE SET
                content_hash = excluded.content_hash, sentiment = excluded.sentiment,
                sentiment_score = excluded.sentiment_score, emotion = excluded.emotion,
                language = excluded.language, model_version = excluded.model_version,
                enriched_at = excluded.enriched_at, {cleared}""",
        rows,
    )


# --- LLM layer --------------------------------------------------------------------------


def pending_llm(
    conn: sqlite3.Connection, scope: Scope, limit: int | None, layer1_version: str
) -> list[sqlite3.Row]:
    """Rows whose layer 1 is current (same version, same content) and that still lack
    the LLM layer, in review_id order."""
    where, params = scope.sql(conn)
    return conn.execute(
        f"""SELECT r.review_id, r.hotel, r.rating, r.title, r.text, r.wrote,
                   r.truncated_in_source
            FROM review r JOIN review_enrichment e USING (review_id)
            WHERE r.text <> '' {where}
              AND e.sentiment IS NOT NULL
              AND e.content_hash = r.content_hash
              AND e.model_version = ?
            ORDER BY r.review_id LIMIT ?""",
        [*params, layer1_version, -1 if limit is None else limit],
    ).fetchall()


def write_llm(
    conn: sqlite3.Connection,
    review_id: int,
    expanded: dict[str, Any],
    staff_normalized: list[str],
    attempts: int,
    full_version: str,
) -> None:
    conn.execute(
        """UPDATE review_enrichment SET
               opinions = ?, staff = ?, incident = ?, says_no_return = ?,
               legal_action = ?, theft = ?, illness = ?, bad_faith = ?,
               returning_guest = ?, with_children = ?, recommends = ?, nights = ?,
               summary = ?, raw_output = ?, llm_attempts = ?,
               model_version = ?, enriched_at = ?
           WHERE review_id = ?""",
        (
            json.dumps(expanded["opinions"], ensure_ascii=False),
            json.dumps(staff_normalized, ensure_ascii=False),
            expanded["incident"],
            int(expanded["says_no_return"]),
            int(expanded["legal_action"]),
            int(expanded["theft"]),
            int(expanded["illness"]),
            int(expanded["bad_faith"]),
            int(expanded["returning_guest"]),
            int(expanded["with_children"]),
            int(expanded["recommends"]),
            expanded["nights"],
            expanded["summary"],
            json.dumps(expanded["raw_output"], ensure_ascii=False),
            attempts,
            full_version,
            now(),
            review_id,
        ),
    )


def hotel_names(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT DISTINCT hotel FROM review ORDER BY 1")]
