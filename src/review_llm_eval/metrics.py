"""Plain-Python metrics (no numpy/sklearn) so that every formula is visible and tested."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PRF:
    precision: float
    recall: float
    f1: float
    support: int  # number of true instances of the class


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def accuracy(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    _check_lengths(y_true, y_pred)
    if not y_true:
        return math.nan
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def binary_prf(tp: int, fp: int, fn: int) -> PRF:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return PRF(precision, recall, f1, tp + fn)


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return math.nan
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _check_lengths(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
