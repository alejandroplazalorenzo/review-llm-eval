"""Shared by the pipeline and the experiments: one review through the model, and the
batch loop (threads for the HTTP calls, results handled in order in the calling thread).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

import requests

from review_llm_eval.client import LLMClient
from review_llm_eval.config import TEMPERATURES
from review_llm_eval.contract import BASE, Variant
from review_llm_eval.data import Review
from review_llm_eval.extract import Extraction, extract

T = TypeVar("T")
R = TypeVar("R")


@dataclass(slots=True)
class Outcome:
    review: Review
    extraction: Extraction | None  # None when the call itself failed
    network_error: str | None = None


def process_one(
    client: LLMClient,
    model: str,
    review: Review,
    variant: Variant = BASE,
    temperatures: Sequence[float] = TEMPERATURES,
) -> Outcome:
    """Never raises for a transport problem: it runs in a worker thread and the error is
    reported back as data, to be counted in order by the calling thread."""
    try:
        return Outcome(review, extract(client, model, review, variant, temperatures))
    except (requests.RequestException, ValueError) as exc:
        return Outcome(review, None, f"network/server error: {exc}")


def run_in_batches(
    items: Sequence[T],
    fn: Callable[[T], R],
    workers: int,
    batch_size: int,
    handle: Callable[[R], bool],
    end_batch: Callable[[int], None],
) -> None:
    """Map ``fn`` over ``items`` with ``workers`` threads, ``batch_size`` items at a time.

    ``handle`` receives every result in input order in the calling thread (so it can write
    to SQLite, whose connections are not shared between threads) and returns ``False`` to
    stop. ``end_batch`` runs after each batch (commit point) with the number of items
    handed out so far.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            stop = False
            for result in pool.map(fn, batch):
                if not handle(result):
                    stop = True
                    break
            end_batch(start + len(batch))
            if stop:
                break
