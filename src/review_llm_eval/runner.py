"""Experiment runner: one model x contract variant x client setting over a list of
reviews, with the same code path as the pipeline (``worker.process_one`` in batches of
``BATCH_SIZE`` over ``workers`` threads).

Writes ``data/experiments/<run_id>.jsonl`` (one record per review, raw answers of every
attempt included; git-ignored because it quotes review text) and
``<run_id>.meta.json`` (settings, environment, wall-clock time). A run resumes where it
stopped unless ``overwrite`` is set; the timing of a resumed run only covers its last
session.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from review_llm_eval import env
from review_llm_eval.client import LLMClient, OllamaClient
from review_llm_eval.config import (
    BATCH_SIZE,
    KEEP_ALIVE,
    MODELS,
    OLLAMA_URL,
    TEMPERATURES,
    WORKERS,
)
from review_llm_eval.contract import BASE, Variant, build_schema
from review_llm_eval.data import Review
from review_llm_eval.jsonl import append_jsonl, iter_jsonl
from review_llm_eval.prompt import build_prompt
from review_llm_eval.worker import Outcome, process_one, run_in_batches


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    experiment: str
    model: str
    subset: str
    variant: Variant = BASE
    workers: int = WORKERS
    base_url: str = OLLAMA_URL
    think: bool | None = False
    keep_alive: str | None = KEEP_ALIVE
    temperatures: tuple[float, ...] = TEMPERATURES
    warm_up: bool = True

    def client(self) -> OllamaClient:
        return OllamaClient(base_url=self.base_url, think=self.think, keep_alive=self.keep_alive)

    def describe(self) -> dict[str, Any]:
        row = asdict(self)
        row["variant"] = asdict(self.variant)
        return row


@dataclass(slots=True)
class Session:
    started_at: str
    n_done: int = 0
    wall_s: float = 0.0
    warm_up_s: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    loaded_after: list[dict[str, Any]] = field(default_factory=list)
    foreign_models: list[str] = field(default_factory=list)
    unloaded_before: list[str] = field(default_factory=list)
    finished_at: str = ""

    @property
    def reviews_per_min(self) -> float:
        return self.n_done / self.wall_s * 60 if self.wall_s else float("nan")


def data_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / f"{run_id}.jsonl"


def meta_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / f"{run_id}.meta.json"


def load_records(out_dir: Path, run_id: str) -> list[dict[str, Any]]:
    path = data_path(out_dir, run_id)
    return list(iter_jsonl(path)) if path.exists() else []


def load_meta(out_dir: Path, run_id: str) -> dict[str, Any]:
    path = meta_path(out_dir, run_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def record_for(spec: RunSpec, outcome: Outcome) -> dict[str, Any]:
    ext = outcome.extraction
    return {
        "run_id": spec.run_id,
        "review_id": outcome.review.review_id,
        "rating": outcome.review.rating,
        "extraction": None if ext is None else ext.to_dict(),
        "network_error": outcome.network_error,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def prepare_server(client: OllamaClient, model: str) -> tuple[list[str], list[str]]:
    """Unload this project's other model if it is still in memory; report anything else
    loaded (another user's model) instead of touching it. Returns (unloaded, foreign)."""
    unloaded, foreign = [], []
    for loaded in client.loaded():
        name = str(loaded.get("name"))
        if name == model:
            continue
        if name in MODELS:
            client.unload(name)
            unloaded.append(name)
        else:
            foreign.append(name)
    return unloaded, foreign


def run(
    spec: RunSpec,
    reviews: Sequence[Review],
    out_dir: Path,
    client: LLMClient | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> Session | None:
    """Run ``spec`` over ``reviews`` (skipping the ones already recorded). Returns the
    session, or ``None`` when there was nothing to do."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = data_path(out_dir, spec.run_id)
    if overwrite:
        path.unlink(missing_ok=True)
        meta_path(out_dir, spec.run_id).unlink(missing_ok=True)
    done = {r["review_id"] for r in iter_jsonl(path)} if path.exists() else set()
    todo = [r for r in reviews if r.review_id not in done]
    if not todo:
        log(f"{spec.run_id}: nothing to do ({len(done)} recorded)")
        return None

    session = Session(started_at=datetime.now(UTC).isoformat(timespec="seconds"))
    real = client is None
    llm: LLMClient = spec.client() if client is None else client
    if real:
        assert isinstance(llm, OllamaClient)
        session.unloaded_before, session.foreign_models = prepare_server(llm, spec.model)
        session.environment = env.collect(llm, spec.model)
    if spec.warm_up:
        start = time.perf_counter()
        llm.generate(spec.model, build_prompt(todo[0], spec.variant), build_schema(spec.variant), 0)
        session.warm_up_s = round(time.perf_counter() - start, 3)

    log(f"{spec.run_id}: {len(todo)} to do ({len(done)} already recorded)")
    t0 = time.perf_counter()

    def handle(outcome: Outcome) -> bool:
        append_jsonl(path, record_for(spec, outcome))
        session.n_done += 1
        return True

    def end_batch(handed_out: int) -> None:
        elapsed = time.perf_counter() - t0
        log(f"  {handed_out}/{len(todo)} ({session.n_done / elapsed * 60:.1f} reviews/min)")

    run_in_batches(
        todo,
        lambda r: process_one(llm, spec.model, r, spec.variant, spec.temperatures),
        spec.workers,
        BATCH_SIZE,
        handle,
        end_batch,
    )
    session.wall_s = round(time.perf_counter() - t0, 3)
    session.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    if real:
        assert isinstance(llm, OllamaClient)
        session.loaded_after = env.loaded_models(llm)
        session.environment["gpu_state_after"] = env.gpu_state()
        others = [m["name"] for m in session.loaded_after if m["name"] != spec.model]
        session.foreign_models = sorted(set(session.foreign_models) | set(others))
    meta = load_meta(out_dir, spec.run_id) or {"spec": spec.describe(), "sessions": []}
    meta["spec"] = spec.describe()
    meta["sessions"].append({**asdict(session), "reviews_per_min": session.reviews_per_min})
    meta_path(out_dir, spec.run_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"{spec.run_id}: {session.n_done} in {session.wall_s:.0f} s")
    return session
