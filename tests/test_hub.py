"""Smoke tests. If the hub cannot be constructed, the deploy is already broken."""

from __future__ import annotations

from convo.hub.app import build_hub
from convo.projects.day01_hub import _run


def test_hub_builds_without_launching():
    demo = build_hub()
    assert demo is not None


def test_day_one_turn_produces_a_trace_a_cost_and_an_eval():
    summary, waterfall, cost, eval_json = _run("I'd like to book an appointment")

    assert "book_appointment" in summary
    assert [row[0] for row in waterfall] == ["nlu", "retrieve", "generate"]
    assert all(row[2] >= 0 for row in waterfall)
    assert "Tokens:" in cost
    assert '"pass_rate": 1.0' in eval_json


def test_day_one_rejects_below_threshold_and_still_returns_a_trace():
    summary, waterfall, _, eval_json = _run("mnghfdsa qwerty")

    assert "fallback" in summary
    assert len(waterfall) == 3
    assert '"verdict": "fail"' in eval_json


def test_empty_input_is_handled_rather_than_crashing():
    summary, waterfall, cost, eval_json = _run("   ")
    assert waterfall == []
    assert cost == ""
    assert eval_json == ""
    assert "Type something" in summary
