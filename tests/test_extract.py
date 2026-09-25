"""One review through the model: validation, retry temperatures, network errors."""

from __future__ import annotations

import pytest
import requests

from helpers import FakeClient, make_output
from review_llm_eval.data import Review
from review_llm_eval.extract import extract
from review_llm_eval.worker import process_one, run_in_batches


def test_valid_first_answer_makes_one_call(reviews: list[Review]) -> None:
    client = FakeClient([make_output()])
    result = extract(client, "qwen3:4b", reviews[0])
    assert result.output == make_output() and result.warning is None
    assert [c["temperature"] for c in client.calls] == [0.0]
    assert result.attempts[0].error is None and result.attempts[0].eval_count == 100


def test_retry_changes_the_temperature(reviews: list[Review]) -> None:
    """Decision test: at temperature 0 the retry would mostly repeat the failure, so the
    second attempt uses 0.4, with the same prompt."""
    client = FakeClient(['{"ops": [', make_output()])
    result = extract(client, "qwen3:4b", reviews[0])
    assert result.output is not None
    temperatures = [c["temperature"] for c in client.calls]
    assert temperatures == [0.0, 0.4]
    assert client.calls[0]["prompt"] == client.calls[1]["prompt"]
    assert "not JSON" in (result.attempts[0].error or "")


def test_invalid_twice_gives_a_warning_and_no_output(reviews: list[Review]) -> None:
    bad = make_output()
    del bad["rsm"]
    client = FakeClient([bad])
    result = extract(client, "qwen3:4b", reviews[0])
    assert result.output is None
    assert result.warning and "after 2 attempts" in result.warning and "rsm" in result.warning
    assert len(client.calls) == 2


def test_network_errors_propagate_from_extract(reviews: list[Review]) -> None:
    client = FakeClient([requests.ConnectionError("down")])
    with pytest.raises(requests.ConnectionError):
        extract(client, "qwen3:4b", reviews[0])


def test_process_one_reports_network_errors_as_data(reviews: list[Review]) -> None:
    outcome = process_one(FakeClient([requests.Timeout("slow")]), "qwen3:4b", reviews[0])
    assert outcome.extraction is None
    assert outcome.network_error and "slow" in outcome.network_error


def test_run_in_batches_keeps_order_and_stops() -> None:
    handled: list[int] = []
    batches: list[int] = []

    def handle(x: int) -> bool:
        handled.append(x)
        return x != 6

    run_in_batches(list(range(10)), lambda x: x * 2, 3, 4, handle, batches.append)
    assert handled == [0, 2, 4, 6]  # in input order, stops at the first False
    assert batches == [4]  # the batch where it stopped is still closed (commit point)
