from __future__ import annotations

from review_llm_eval.examples import pseudonymise


def test_staff_names_and_other_capitalised_words_are_masked() -> None:
    text = "Gracias a Marta y al animador Pedro por todo, y a JOSÉ del bar."
    out = pseudonymise(text, ["Marta", "José"])
    assert out is not None
    assert "Marta" not in out and "Pedro" not in out and "JOSÉ" not in out
    assert out.startswith("Gracias a [staff")  # the first word of a sentence stays
    assert pseudonymise(None, []) is None


def test_placeholders_are_consistent_and_quotes_are_cut() -> None:
    names = ["José", "JOSE", "Marta"]
    out = pseudonymise("Gracias a Marta y a José por todo el servicio", names, limit=30)
    assert out == "Gracias a [staff 2] y a..."
    assert pseudonymise("josé", names) == "[staff 1]"
