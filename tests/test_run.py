from __future__ import annotations

import json
from pathlib import Path

from helpers import FakeClient, make_output
from review_llm_eval.classify import ClassificationResult
from review_llm_eval.client import DEFAULT_OPTIONS
from review_llm_eval.config import model_slug
from review_llm_eval.data import Review
from review_llm_eval.run import output_path, run_batch
from review_llm_eval.schema import build_schema


def test_output_path_uses_slug_and_tag(tmp_path: Path) -> None:
    assert model_slug("qwen2.5:7b-instruct") == "qwen2.5_7b-instruct"
    assert output_path("gemma3:4b", root=tmp_path) == tmp_path / "gemma3_4b.jsonl"
    assert output_path("gemma3:4b", "rerun", tmp_path) == tmp_path / "gemma3_4b__rerun.jsonl"


def test_run_batch_returns_one_result_per_review(reviews: list[Review]) -> None:
    answer = json.dumps(make_output("positive"))
    client = FakeClient([answer] * len(reviews))
    collected: list[ClassificationResult] = []
    run_batch(client, "m", reviews, build_schema(), DEFAULT_OPTIONS, 3, collected.append)
    assert sorted(r.review_id for r in collected) == sorted(r.review_id for r in reviews)
    assert all(r.status == "ok" for r in collected)
