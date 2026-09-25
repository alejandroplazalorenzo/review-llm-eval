"""Deterministic post-processing: expansion, de-duplication, quotes and staff names."""

from __future__ import annotations

from helpers import make_output
from review_llm_eval.contract import BASE, Variant
from review_llm_eval.postprocess import (
    clean_staff,
    expand,
    fold,
    hotel_veto,
    is_filler,
    name_in_text,
    normalize_name,
    quote_support,
    veto_for,
)

SOURCE = "Llegamos y el aire acondicionado no funcionaba. Marta fue muy amable con nosotros."


def test_expand_translates_to_long_names() -> None:
    e = expand(make_output(), BASE, SOURCE)
    assert e["opinions"][0] == {
        "topic": "room",
        "polarity": "negative",
        "quote": "el aire acondicionado no funcionaba",
    }
    assert e["incident"] == "complaint_ignored"
    assert e["says_no_return"] is True and e["with_children"] is True
    assert e["nights"] == 7 and e["model_language"] == "spanish"
    assert e["raw_output"] == make_output()  # kept for re-mapping without the GPU


def test_incident_none_becomes_null() -> None:
    assert expand(make_output(inc="none"), BASE)["incident"] is None


def test_repeated_topic_polarity_pairs_are_collapsed() -> None:
    ops = [
        {"t": "buf", "p": "neg", "lit": "la comida fría"},
        {"t": "buf", "p": "neg", "lit": "otra cita"},
        {"t": "buf", "p": "pos", "lit": "el postre"},
    ]
    opinions = expand(make_output(ops=ops), BASE)["opinions"]
    assert [(o["topic"], o["polarity"]) for o in opinions] == [
        ("food", "negative"),
        ("food", "positive"),
    ]
    assert opinions[0]["quote"] == "la comida fría"  # the first one wins


def test_filler_and_invented_quotes_are_dropped_but_the_opinion_stays() -> None:
    ops = [
        {"t": "hab", "p": "neg", "lit": "No se menciona."},
        {"t": "atn", "p": "pos", "lit": "trato del personal: amabilidad, atención, rapidez"},
    ]
    opinions = expand(make_output(ops=ops), BASE, SOURCE)["opinions"]
    assert [o["quote"] for o in opinions] == [None, None]
    assert [o["topic"] for o in opinions] == ["room", "staff_service"]


def test_quote_support() -> None:
    assert quote_support("el AIRE  acondicionado", SOURCE) == 1.0  # case and spacing
    assert quote_support("Marta fue muy amable", SOURCE) == 1.0
    assert quote_support("Marta fue amabilísima con todos", SOURCE) < 0.5
    assert quote_support("", SOURCE) == 0.0
    assert quote_support("aire acondicionado roto", SOURCE) == 2 / 3


def test_is_filler() -> None:
    assert is_filler("N/A") and is_filler("No se menciona.") and is_filler("")
    assert not is_filler("la playa")


def test_staff_must_be_written_in_the_review() -> None:
    assert clean_staff(["Marta", "Eneko"], SOURCE) == ["Marta"]


def test_staff_match_is_whole_word() -> None:
    assert not name_in_text("Ana", "una banana muy rica")
    assert name_in_text("José Luis", "gracias a jose  luis por todo")


def test_roles_and_generic_words_are_not_names() -> None:
    names = ["el camarero", "Mayordomo", "Marta", "todos", "Recepcionista"]
    assert clean_staff(names, None) == ["Marta"]


def test_hotel_chain_and_place_names_are_vetoed() -> None:
    veto = hotel_veto(["Iberostar Dominicana", "Iberostar Punta Cana", "Serena Village"])
    assert "iberostar" in veto  # shared by two hotels: a chain
    assert "serena" not in veto  # one hotel only: could be a person
    text = "Gracias a Serena del bar y a todo Iberostar en Punta Cana"
    assert clean_staff(["Iberostar", "Punta Cana", "Serena"], text, veto) == ["Serena"]
    # ...but not in a review of the hotel called Serena Village
    assert clean_staff(["Serena"], text, veto_for("Serena Village", veto)) == []


def test_staff_deduplicated_ignoring_case_and_accents() -> None:
    assert clean_staff(["José", "JOSE", "jose"], "José nos atendió") == ["José"]
    assert normalize_name("  José   Ángel ") == "jose angel"


def test_expand_long_variant() -> None:
    raw = {
        "opinions": [{"topic": "beach", "polarity": "positive", "quote": "playa"}],
        "staff": [],
        "incident": "none",
        "language": "spanish",
        "summary": "Bien.",
        "nights": None,
    }
    e = expand(raw, Variant(keys="long"))
    assert e["opinions"][0]["topic"] == "beach" and e["incident"] is None


def test_fold() -> None:
    assert fold("  Ñandú   ÁRBOL ") == "nandu arbol"
