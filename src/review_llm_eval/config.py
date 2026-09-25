"""Project-wide constants: models, Ollama settings, operational limits and paths.

The Ollama settings mirror the review-enrichment pipeline I run at work (see README,
"Decisions from my production system"); the numbers that justify them are re-measured
on public data by the experiments in ``experiments.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --- Ollama ---------------------------------------------------------------------------

OLLAMA_URL: Final = "http://127.0.0.1:11434"
"""127.0.0.1, not localhost: on Windows ``localhost`` may resolve to ``::1`` first and pay
a connection timeout on every call when the server only listens on IPv4 (experiment E7)."""

MAIN_MODEL: Final = "qwen3:4b"
COMPARISON_MODEL: Final = "qwen3:8b"
MODELS: Final = (MAIN_MODEL, COMPARISON_MODEL)

EXPECTED_DIGESTS: Final[dict[str, str]] = {
    # 12-hex prefix of the local Ollama digest. Recorded with every run; a different
    # digest means different weights, so the numbers in the README no longer apply.
    MAIN_MODEL: "359d7dd4bcda",
    COMPARISON_MODEL: "500a1f067a9f",
}
MODEL_NOTES: Final[dict[str, str]] = {
    MAIN_MODEL: (
        "Qwen3-4B, 2507 'thinking' release (declares 262,144 context; its chat template "
        "always opens <think> and has no branch for think=false)"
    ),
    COMPARISON_MODEL: (
        "Qwen3-8B, original release (declares 40,960 context; its template adds "
        "/no_think and an empty <think></think> when think=false)"
    ),
}

WORKERS: Final = 4
"""Client threads. Only useful if the server runs with OLLAMA_NUM_PARALLEL >= WORKERS."""
TIMEOUT_S: Final = 300.0
MAX_INPUT_CHARS: Final = 4000
"""Kept for parity with production; never triggers here (the dataset caps texts at 790)."""
KEEP_ALIVE: Final = "30m"
NUM_CTX: Final = 4096
NUM_PREDICT: Final = 1024
TEMPERATURES: Final[tuple[float, ...]] = (0.0, 0.4)
"""First attempt deterministic, the retry with some randomness (experiment E4)."""

BATCH_SIZE: Final = 25
"""Rows per commit: a killed run resumes from the last committed batch."""
MAX_CONSECUTIVE_NETWORK_ERRORS: Final = 5
MAX_WARNINGS_SHOWN: Final = 20

SERVER_ENV_VARS: Final = ("OLLAMA_NUM_PARALLEL", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE")

# --- Data -------------------------------------------------------------------------------

DATASET_ID: Final = "beltrewilton/punta-cana-spanish-reviews"
DATASET_PARQUET_API: Final = f"https://huggingface.co/api/datasets/{DATASET_ID}/parquet"

SEED: Final = 42
"""Seed for sampling only. No seed is ever sent to the model (production does not use one)."""
PER_RATING: Final = 40  # 5 ratings x 40 = 200 reviews for the model comparison (E1)
GOLD_TARGET: Final = 100  # reviews queued for human labelling (20 per rating)

DATA_DIR: Final = Path("data")
RAW_PARQUET: Final = DATA_DIR / "raw" / "punta_cana_train.parquet"
SAMPLE_PATH: Final = DATA_DIR / "sample.jsonl"
STORE_PATH: Final = DATA_DIR / "reviews.sqlite"
EXPERIMENTS_DIR: Final = DATA_DIR / "experiments"
RESULTS_DIR: Final = Path("results")
GOLD_PATH: Final = Path("gold") / "gold.jsonl"


def model_slug(model: str) -> str:
    """File-system friendly name for a model tag, e.g. ``qwen3:4b`` -> ``qwen3_4b``."""
    return model.replace(":", "_").replace("/", "_")
