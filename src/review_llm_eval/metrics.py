"""Plain-Python metrics (no numpy/sklearn) so that every formula is visible and tested."""

from __future__ import annotations

import math
from collections import Counter
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


def confusion_matrix(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels_true: Sequence[Hashable],
    labels_pred: Sequence[Hashable],
) -> list[list[int]]:
    """Counts with rows = true labels and columns = predicted labels.

    Pairs whose label is not listed are ignored.
    """
    _check_lengths(y_true, y_pred)
    row_index = {label: i for i, label in enumerate(labels_true)}
    col_index = {label: j for j, label in enumerate(labels_pred)}
    matrix = [[0] * len(labels_pred) for _ in labels_true]
    for t, p in zip(y_true, y_pred, strict=True):
        if t in row_index and p in col_index:
            matrix[row_index[t]][col_index[p]] += 1
    return matrix


def accuracy(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    _check_lengths(y_true, y_pred)
    if not y_true:
        return math.nan
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def prf_for_label(y_true: Sequence[Hashable], y_pred: Sequence[Hashable], label: Hashable) -> PRF:
    """One-vs-rest precision, recall and F1 for ``label``.

    Predictions outside the true label set (e.g. "mixed" against a binary proxy) count
    as false negatives for the true class and never as true positives.
    """
    _check_lengths(y_true, y_pred)
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
    return binary_prf(tp, fp, fn)


def binary_prf(tp: int, fp: int, fn: int) -> PRF:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return PRF(precision, recall, f1, tp + fn)


def macro_f1(
    y_true: Sequence[Hashable], y_pred: Sequence[Hashable], labels: Sequence[Hashable]
) -> float:
    """Unweighted mean of per-label F1 over ``labels``."""
    if not labels:
        return math.nan
    return sum(prf_for_label(y_true, y_pred, label).f1 for label in labels) / len(labels)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa between two raters over the same items.

    Returns ``nan`` when expected agreement is 1 (both raters constant and equal),
    where kappa is undefined.
    """
    _check_lengths(a, b)
    n = len(a)
    if n == 0:
        return math.nan
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[k] * count_b[k] for k in count_a.keys() | count_b.keys()) / (n * n)
    if math.isclose(expected, 1.0):
        return math.nan
    return (observed - expected) / (1 - expected)


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
