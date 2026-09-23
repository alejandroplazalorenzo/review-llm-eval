"""Small helpers to write aggregate results as CSV and Markdown."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

Cell = str | int | float | bool | None


def fmt(value: Cell, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Cell]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([fmt(v, 4) for v in row])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Cell]], digits: int = 3) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(fmt(v, digits) for v in row) + " |" for row in rows)
    return "\n".join(lines)
