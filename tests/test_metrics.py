from __future__ import annotations

import math

import pytest

from review_llm_eval.metrics import (
    accuracy,
    binary_prf,
    cohen_kappa,
    confusion_matrix,
    macro_f1,
    percentile,
    prf_for_label,
)


def test_confusion_matrix_rows_true_cols_pred() -> None:
    y_true = ["neg", "neg", "pos", "pos", "pos"]
    y_pred = ["neg", "mixed", "pos", "neg", "pos"]
    m = confusion_matrix(y_true, y_pred, ["neg", "pos"], ["pos", "neg", "mixed"])
    assert m == [[0, 1, 1], [2, 1, 0]]


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


def test_prediction_outside_proxy_labels_counts_as_miss() -> None:
    # "mixed" is never a true label in the binary proxy: it lowers recall of the true
    # class and does not create a false positive for any proxy class.
    y_true = ["negative", "negative", "positive", "positive"]
    y_pred = ["negative", "mixed", "positive", "positive"]
    neg = prf_for_label(y_true, y_pred, "negative")
    pos = prf_for_label(y_true, y_pred, "positive")
    assert (neg.precision, neg.recall) == (1.0, 0.5)
    assert (pos.precision, pos.recall) == (1.0, 1.0)
    assert macro_f1(y_true, y_pred, ["negative", "positive"]) == pytest.approx((2 / 3 + 1) / 2)


def test_cohen_kappa_textbook_example() -> None:
    # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15.
    # p_o = 0.7, p_e = 0.5*0.6 + 0.5*0.4 = 0.5  ->  kappa = 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_perfect_and_chance() -> None:
    assert cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == pytest.approx(1.0)
    # observed agreement equal to chance agreement -> 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_cohen_kappa_undefined_when_both_constant() -> None:
    assert math.isnan(cohen_kappa([True] * 5, [True] * 5))
    assert math.isnan(cohen_kappa([], []))


def test_percentile_matches_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(8.8)  # numpy.percentile gives 8.8
    assert percentile([5.0], 95) == 5.0
    assert math.isnan(percentile([], 50))
    with pytest.raises(ValueError):
        percentile(values, 101)
