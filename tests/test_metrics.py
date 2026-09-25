from __future__ import annotations

import math

import pytest

from review_llm_eval.metrics import accuracy, binary_prf, mean, percentile


def test_accuracy() -> None:
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "x"]) == 0.5
    assert math.isnan(accuracy([], []))
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_binary_prf_known_values() -> None:
    prf = binary_prf(tp=6, fp=2, fn=4)
    assert prf.precision == pytest.approx(0.75)
    assert prf.recall == pytest.approx(0.6)
    assert prf.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert prf.support == 10


def test_binary_prf_zero_division_is_zero() -> None:
    prf = binary_prf(0, 0, 0)
    assert (prf.precision, prf.recall, prf.f1) == (0.0, 0.0, 0.0)


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(8.8)  # numpy.percentile gives 8.8
    assert percentile([5.0], 95) == 5.0
    assert math.isnan(percentile([], 50))
    with pytest.raises(ValueError):
        percentile(values, 101)


def test_mean_of_nothing_is_nan() -> None:
    assert mean([1.0, 2.0]) == 1.5
    assert math.isnan(mean([]))
