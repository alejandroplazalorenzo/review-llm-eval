"""Layer 1 of the enrichment: sentiment and emotion (pysentimiento / RoBERTuito) plus the
deterministic language. No LLM involved.

The overall sentiment of a review comes from here, not from the LLM: a dedicated
classifier is cheaper, runs on CPU and does not change when the prompt changes.

Version contract on ``review_enrichment.model_version`` (one column for both layers)::

    LAYER1_VERSION                          layer 1 done, LLM layer pending
    LAYER1_VERSION + "+" + <model>/<prompt> both layers done

Rewriting layer 1 for a row (new text, or ``--reprocess``) empties every LLM column of
that row: the old LLM output belongs to another text or version and must not survive
until the LLM layer reaches the row again.

``pysentimiento`` is an optional extra (``pip install -e ".[layer1]"``); nothing here
imports it until an analyzer is actually needed.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from review_llm_eval import store
from review_llm_eval.langdetect import DETECTOR_VERSION, detect

LAYER1_VERSION = f"pysentimiento-robertuito+{DETECTOR_VERSION}/v1"
BATCH_SIZE = 64
MAX_CHARS = 2000  # RoBERTuito reads 128 tokens; cutting earlier saves tokenisation


@dataclass(frozen=True, slots=True)
class Layer1Result:
    sentiment: str  # POS / NEU / NEG
    sentiment_score: float  # probability of the predicted label
    emotion: str


class Analyzers(Protocol):
    def predict(self, texts: Sequence[str]) -> list[Layer1Result]: ...


class PysentimientoAnalyzers:
    """Sentiment + emotion analyzers for Spanish (downloaded from Hugging Face once)."""

    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
        try:
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
        try:
            from pysentimiento import create_analyzer
        except ImportError as exc:
            raise SystemExit(
                f'layer 1 needs pysentimiento ({exc}); install it with: pip install -e ".[layer1]"'
            ) from exc
        self._sentiment = create_analyzer(task="sentiment", lang="es")
        self._emotion = create_analyzer(task="emotion", lang="es")

    def predict(self, texts: Sequence[str]) -> list[Layer1Result]:
        sentiments = self._sentiment.predict(list(texts))
        emotions = self._emotion.predict(list(texts))
        return [
            Layer1Result(s.output, float(s.probas[s.output]), e.output)
            for s, e in zip(sentiments, emotions, strict=True)
        ]


def layer1_text(title: str, text: str) -> str:
    return (f"{title}. {text}" if title else text)[:MAX_CHARS]


def run_layer1(
    conn: sqlite3.Connection,
    analyzers: Analyzers | None,
    scope: store.Scope,
    reprocess: bool,
    full_version: str,
    limit: int | None = None,
    load_analyzers: Callable[[], Analyzers] | None = None,
) -> int:
    """Process every row pending layer 1 in ``scope`` (commit per batch). Returns the
    number of rows written. Analyzers are only loaded if there is something to do."""
    done = 0
    while limit is None or done < limit:
        size = BATCH_SIZE if limit is None else min(BATCH_SIZE, limit - done)
        rows = store.pending_layer1(conn, scope, size, reprocess, LAYER1_VERSION, full_version)
        if not rows:
            break
        if analyzers is None:
            analyzers = (load_analyzers or PysentimientoAnalyzers)()
        results = analyzers.predict([layer1_text(r["title"], r["text"]) for r in rows])
        store.write_layer1(
            conn,
            [
                (
                    r["review_id"],
                    r["content_hash"],
                    res.sentiment,
                    res.sentiment_score,
                    res.emotion,
                    detect(f"{r['title']}\n{r['text']}"),
                    LAYER1_VERSION,
                )
                for r, res in zip(rows, results, strict=True)
            ],
        )
        conn.commit()
        done += len(rows)
    return done
