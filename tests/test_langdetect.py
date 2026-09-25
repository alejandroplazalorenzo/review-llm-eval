from __future__ import annotations

import pytest

from review_llm_eval.langdetect import detect


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("La habitación estaba sucia pero el personal fue muy amable con los niños.", "es"),
        ("We had a great time and the staff were very friendly with our kids.", "en"),
        ("O quarto era ótimo, muito limpo, e os funcionários são gentis. Não voltaria?", "pt"),
        ("Das Zimmer war sehr sauber und wir sind mit dem Essen nicht zufrieden.", "de"),
        ("Todo perfecto", "es"),  # no function word: defaults to Spanish
        ("", "es"),
        (None, "es"),
    ],
)
def test_detect(text: str | None, lang: str) -> None:
    assert detect(text) == lang


def test_spanish_with_a_few_english_words_stays_spanish() -> None:
    text = "El hotel es top, el staff muy friendly y la comida del buffet, the best. Volveremos."
    assert detect(text) == "es"
