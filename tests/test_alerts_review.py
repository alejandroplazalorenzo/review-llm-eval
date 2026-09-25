from __future__ import annotations

from pathlib import Path

from review_llm_eval.alerts_review import load_verdicts, review_loop, summarize

HITS = [
    (1, "theft", "Robo", "Nos robaron el móvil"),
    (2, "theft", "Bien", "Todo bien"),
    (3, "legal_action", "Mal", "Pondremos una denuncia"),
]


def test_review_loop_records_verdicts_and_summary_says_at_least(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    answers = iter(["y", "maybe", "n", "q"])
    added = review_loop(HITS, path, "AP", lambda _: next(answers), lambda *_: None)
    assert added == 2
    verdicts = load_verdicts(path)
    assert verdicts == {(1, "theft"): True, (2, "theft"): False}
    rows = {r[0]: r for r in summarize(HITS, verdicts)}
    assert rows["theft"][1:] == [2, 2, "at least 1"]
    assert rows["legal_action"][1:] == [1, 0, "pending"]
