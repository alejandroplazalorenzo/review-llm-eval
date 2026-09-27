import pytest

from review_llm_eval import rules


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Nunca más volveremos a este hotel.", True),
        ("No lo recomiendo, fue un desastre.", True),
        ("Desaconsejo este sitio.", True),
        ("La comida fría y el servicio lento. Muy decepcionados.", False),
        ("Todo perfecto, gracias.", False),
    ],
)
def test_states_no_return(text: str, expected: bool) -> None:
    assert rules.states_no_return(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Lo recomiendo al 100%.", True),
        ("Hotel muy recomendable para familias.", True),
        ("No dudaría en recomendarlo.", True),
        ("No lo recomiendo a nadie.", False),
        ("Excelente hotel, todo genial, volveremos.", False),
        ("La playa bonita. No recomiendo el buffet italiano.", False),
    ],
)
def test_states_recommendation(text: str, expected: bool) -> None:
    assert rules.states_recommendation(text) is expected


def _expanded(**flags: bool) -> dict[str, object]:
    return {"says_no_return": False, "recommends": False, "theft": False, **flags}


def test_apply_drops_flags_the_text_cannot_support() -> None:
    out = rules.apply(
        _expanded(says_no_return=True, recommends=True), "Todo genial, un servicio de diez."
    )
    assert out["says_no_return"] is False
    assert out["recommends"] is False


def test_apply_keeps_flags_the_text_states() -> None:
    out = rules.apply(_expanded(says_no_return=True), "Nunca más, no volvemos.")
    assert out["says_no_return"] is True
    out = rules.apply(_expanded(recommends=True), "Lo recomiendo sin duda.")
    assert out["recommends"] is True


def test_apply_never_adds_a_flag_and_leaves_other_fields_alone() -> None:
    original = _expanded(theft=True)
    out = rules.apply(original, "No lo recomiendo. Nos robaron la maleta.")
    assert out["says_no_return"] is False
    assert out["theft"] is True
    assert original == _expanded(theft=True)


def test_apply_without_text_is_a_no_op() -> None:
    original = _expanded(says_no_return=True, recommends=True)
    assert rules.apply(original, None) is original
