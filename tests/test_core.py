from __future__ import annotations

import pytest

from convo.core import costs, evals, registry, tracing


class TestTracing:
    def test_spans_are_recorded_in_order_with_offsets(self):
        with tracing.trace() as active:
            with tracing.span("first"):
                pass
            with tracing.span("second", k=3):
                pass

        assert [s.name for s in active.spans] == ["first", "second"]
        assert active.spans[1].start_ms >= active.spans[0].start_ms
        assert active.spans[1].attrs == {"k": 3}
        assert all(s.duration_ms >= 0 for s in active.spans)

    def test_nested_spans_record_their_parent(self):
        with tracing.trace() as active:
            with tracing.span("outer") as outer:
                with tracing.span("inner"):
                    pass

        inner = next(s for s in active.spans if s.name == "inner")
        assert inner.parent == outer.span_id

    def test_span_outside_a_trace_does_not_raise(self):
        with tracing.span("orphan"):
            pass
        assert tracing.current_trace() is None

    def test_waterfall_rows_are_plottable(self):
        with tracing.trace() as active:
            with tracing.span("stt"):
                pass
        rows = active.waterfall()
        assert len(rows) == 1
        name, offset, duration = rows[0]
        assert name == "stt"
        assert offset >= 0 and duration >= 0


class TestCostMeter:
    def test_unpriced_models_count_tokens_but_report_no_cost(self):
        meter = costs.CostMeter()
        call = meter.record("mystery", 1000, 500)

        assert call.usd is None
        assert call.total_tokens == 1500
        assert meter.total_tokens == 1500
        assert meter.unpriced == {"mystery"}

    def test_require_total_usd_refuses_to_guess(self):
        meter = costs.CostMeter()
        meter.record("mystery", 10, 10)
        with pytest.raises(costs.UnpricedModel):
            meter.require_total_usd()

    def test_registered_price_is_applied_per_million_tokens(self):
        meter = costs.CostMeter()
        meter.register_price("m", input_per_mtok=2.0, output_per_mtok=10.0)
        meter.record("m", 1_000_000, 1_000_000)

        assert meter.require_total_usd() == pytest.approx(12.0)

    def test_by_model_aggregates(self):
        meter = costs.CostMeter()
        meter.register_price("m", 1.0, 1.0)
        meter.record("m", 100, 100)
        meter.record("m", 100, 100)

        row = meter.by_model()["m"]
        assert row["calls"] == 2
        assert row["input_tokens"] == 200

    def test_price_book_loads_from_toml(self, tmp_path):
        book = tmp_path / "prices.toml"
        book.write_text(
            '[models."m"]\ninput_per_mtok = 3.0\noutput_per_mtok = 15.0\n', encoding="utf-8"
        )
        meter = costs.CostMeter.from_toml(book)
        assert meter.prices["m"].output_per_mtok == 15.0


class TestEvalRun:
    def _run(self, outcomes: dict[str, bool], suite: str = "s") -> evals.EvalRun:
        run = evals.EvalRun(suite=suite)
        for case_id, ok in outcomes.items():
            run.add(evals.CaseResult.from_assertions(case_id, [evals.Assertion("check", ok)]))
        return run

    def test_pass_rate_and_failures(self):
        run = self._run({"a": True, "b": False, "c": True, "d": True})
        assert run.pass_rate == 0.75
        assert [c.case_id for c in run.failures] == ["b"]

    def test_empty_run_is_zero_not_a_crash(self):
        assert evals.EvalRun(suite="empty").pass_rate == 0.0

    def test_gate_compares_against_threshold(self):
        run = self._run({"a": True, "b": True})
        assert run.gate(1.0) is True
        assert run.gate(1.01) is False

    def test_compare_names_regressions_and_fixes(self):
        before = self._run({"a": True, "b": False, "gone": True})
        after = self._run({"a": False, "b": True, "new": True})

        diff = after.compare(before)
        assert diff.regressions == ["a"]
        assert diff.fixes == ["b"]
        assert diff.added == ["new"]
        assert diff.removed == ["gone"]
        assert diff.clean is False

    def test_round_trips_through_json(self, tmp_path):
        run = self._run({"a": True, "b": False})
        run.cases[0].transcript = [{"role": "user", "content": "hi"}]
        path = run.save(tmp_path / "runs" / "run.json")

        loaded = evals.EvalRun.load(path)
        assert loaded.run_id == run.run_id
        assert loaded.pass_rate == run.pass_rate
        assert loaded.cases[0].transcript == [{"role": "user", "content": "hi"}]
        assert loaded.cases[0].assertions[0].name == "check"


class TestRegistry:
    def test_the_plan_holds_twenty_one_days_across_three_weeks(self):
        all_days = registry.projects()
        assert len(all_days) == 21
        assert [p.day for p in all_days] == list(range(1, 22))
        assert set(registry.by_week()) == {1, 2, 3}

    def test_day_one_is_shipped_once_projects_are_imported(self):
        import convo.projects  # noqa: F401

        assert registry.get(1).status == "shipped"
        assert registry.get(1).builder is not None

    def test_planned_days_have_a_brief_and_no_builder(self):
        planned = [p for p in registry.projects() if p.status == "planned"]
        assert planned, "day 1 should be the only shipped day at this point"
        assert all(p.brief.strip() for p in planned)
        assert all(p.builder is None for p in planned)

    def test_shipping_an_unknown_day_is_an_error(self):
        with pytest.raises(KeyError):
            registry.ship(99, lambda: None)
