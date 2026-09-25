"""Test helpers: a schema-valid model output and a scripted fake LLM client."""

from __future__ import annotations

import copy
import json
from typing import Any

from review_llm_eval.client import Generation


def make_output(**overrides: Any) -> dict[str, Any]:
    """A schema-valid answer in the production (short-key) contract."""
    out: dict[str, Any] = {
        "ops": [
            {"t": "hab", "p": "neg", "lit": "el aire acondicionado no funcionaba"},
            {"t": "atn", "p": "pos", "lit": "Marta fue muy amable"},
        ],
        "emp": ["Marta"],
        "inc": "ignored",
        "nov": True,
        "leg": False,
        "rob": False,
        "enf": False,
        "fra": False,
        "ret": False,
        "kid": True,
        "recom": False,
        "noc": 7,
        "idi": "es",
        "rsm": "El aire acondicionado falló y nadie lo arregló.",
    }
    out.update(copy.deepcopy(overrides))
    return out


class FakeClient:
    """Stand-in for ``OllamaClient``: returns (or raises) the scripted items in order and
    records every call. A dict is serialised to JSON; a string is returned as is."""

    def __init__(self, script: list[Any], done_reason: str = "stop") -> None:
        self.script = list(script)
        self.done_reason = done_reason
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, model: str, prompt: str, schema: dict[str, Any], temperature: float
    ) -> Generation:
        self.calls.append(
            {"model": model, "prompt": prompt, "schema": schema, "temperature": temperature}
        )
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return Generation(
            response=text,
            done_reason=self.done_reason,
            eval_count=100,
            prompt_eval_count=900,
            total_duration_s=1.0,
            eval_duration_s=0.8,
        )
