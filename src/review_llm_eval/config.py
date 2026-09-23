"""Project-wide constants: taxonomy, label sets, defaults and paths."""

from __future__ import annotations

from pathlib import Path
from typing import Final

OLLAMA_URL: Final = "http://127.0.0.1:11434"
"""127.0.0.1 on purpose: on the development machine (Windows 11) a GET to
``http://localhost:11434/api/version`` took ~2.07 s versus ~0.015 s for 127.0.0.1."""

DEFAULT_MODELS: Final = ("qwen2.5:7b-instruct", "gemma3:4b")

ASPECTS: Final[tuple[str, ...]] = (
    "staff",
    "cleanliness",
    "room",
    "food",
    "pool_beach",
    "location",
    "value",
    "check_in",
    "noise",
    "safety",
    "entertainment",
    "other",
)

OVERALL_SENTIMENTS: Final[tuple[str, ...]] = ("positive", "neutral", "negative", "mixed")
ASPECT_SENTIMENTS: Final[tuple[str, ...]] = ("positive", "neutral", "negative", "mixed")

# Star rating -> proxy sentiment. 3 stars has no proxy label and is reported apart.
PROXY_LABELS: Final[tuple[str, ...]] = ("negative", "positive")

SEED: Final = 42
PER_RATING: Final = 70  # 5 ratings x 70 = 350 reviews
GOLD_TARGET: Final = 200  # reviews queued for human labelling (40 per rating)

DATASET_ID: Final = "beltrewilton/punta-cana-spanish-reviews"
DATASET_PARQUET_API: Final = f"https://huggingface.co/api/datasets/{DATASET_ID}/parquet"

DATA_DIR: Final = Path("data")
RAW_PARQUET: Final = DATA_DIR / "raw" / "punta_cana_train.parquet"
SAMPLE_PATH: Final = DATA_DIR / "sample.jsonl"
OUTPUTS_DIR: Final = DATA_DIR / "outputs"
RESULTS_DIR: Final = Path("results")
GOLD_PATH: Final = Path("gold") / "gold.jsonl"


def model_slug(model: str) -> str:
    """File-system friendly name for a model tag, e.g. ``qwen2.5:7b-instruct`` ->
    ``qwen2.5_7b-instruct``."""
    return model.replace(":", "_").replace("/", "_")
