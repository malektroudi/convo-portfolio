"""Day 16: that the comparison is fair, and that a quota failure is not a score.

The point of rebuilding the bot a third time is to compare three
architectures, which only works if the things that are supposed to be
constant actually are. So the tests here are mostly about what is *shared*:
day 5's tools, day 6's scorers, day 6's suite.

The other half is about the free tier. A daily quota that runs out mid-suite
produces a partial run, and a partial run reported as a complete one is the
kind of number this repository exists to avoid.
"""

from __future__ import annotations

import json
import sys
from importlib.abc import MetaPathFinder

import pytest

from convo.projects.day16_adk_agent import build
from convo.projects.day16_adk_agent import model as mdl

# No `importorskip` here, and that is the point: none of day 16's modules
# imports ADK at module level, so all of this runs in CI, where only the dev
# extra is installed. `TestADKStaysOutOfTheServingPath` is what holds that
# true.


class TestTheComparisonIsFair:
    def test_it_uses_day_fives_tools_not_copies(self):
        """A reimplementation would let a rules difference look like an
        architecture difference."""
        from convo.projects.day05_tool_agent.tools import book_table, get_policy
        from convo.projects.day16_adk_agent import agent

        assert agent.book_table is book_table
        assert agent.get_policy is get_policy

    def test_it_uses_day_fives_system_prompt(self):
        from convo.projects.day05_tool_agent.tools import system_prompt
        from convo.projects.day16_adk_agent import agent

        assert agent.system_prompt is system_prompt

    def test_it_scores_with_day_sixes_harness(self):
        from convo.projects.day06_eval_harness.harness import run_suite

        assert build.run_suite is run_suite

    def test_it_runs_day_sixes_suite_and_reference(self):
        from convo.projects.day06_eval_harness.suite import REF, SUITE

        assert build.SUITE is SUITE
        assert build.REF is REF

    def test_it_normalises_the_booking_the_same_way(self):
        from convo.projects.day05_tool_agent.scenarios import normalise_booking

        assert build.normalise_booking is normalise_booking

    def test_the_booking_is_read_from_the_tool_not_the_prose(self):
        """Day 5's lesson: a model will describe a reservation it never made.

        `Booking.fields` is only ever written inside the tool wrapper, on a
        successful call, so a confident reply cannot create one.
        """
        import inspect

        from convo.projects.day16_adk_agent import agent

        source = inspect.getsource(agent.build_tools)
        assert "booking.fields = {" in source
        assert source.count("booking.fields") == 1
        outer = inspect.getsource(agent).replace(source, "")
        assert "booking.fields =" not in outer


class TestTheBookingTool:
    """The wrapper must not soften day 5's rules."""

    @pytest.fixture
    def harness(self):
        from convo.projects.day06_eval_harness.suite import REF
        from convo.projects.day16_adk_agent.agent import Booking, build_tools

        booking = Booking()
        book, policy = build_tools(REF, booking)
        return book, policy, booking, REF

    def test_a_good_booking_is_captured(self, harness):
        book, _, booking, ref = harness
        when = ref.today.replace(day=min(ref.today.day, 28))
        result = book(2, when.isoformat(), "20:00", "Chen")
        if not result["ok"]:
            pytest.skip(f"reference calendar rejected the date: {result['error']}")
        assert booking.fields["party_size"] == 2
        assert booking.fields["name"] == "Chen"

    def test_a_refused_booking_captures_nothing(self, harness):
        book, _, booking, _ = harness
        result = book(2, "1999-01-01", "20:00", "Chen")
        assert not result["ok"]
        assert booking.fields is None
        assert booking.failures == 1

    def test_an_unreadable_time_is_refused(self, harness):
        book, _, booking, ref = harness
        assert not book(2, ref.today.isoformat(), "half past eight", "Chen")["ok"]
        assert booking.fields is None

    def test_every_call_is_recorded_either_way(self, harness):
        book, _, booking, _ = harness
        book(2, "1999-01-01", "20:00", "Chen")
        book(2, "1999-01-02", "20:00", "Chen")
        assert len(booking.tool_calls) == 2
        assert all(c["tool"] == "book_table" for c in booking.tool_calls)


class TestTheModelResolver:
    def test_it_prefers_a_model_with_a_workable_daily_quota(self):
        """20 requests a day will not run a 32-scenario suite."""
        assert mdl.PREFERENCE[0].endswith("lite")

    def test_it_excludes_models_that_cannot_do_this_task(self):
        import inspect

        source = inspect.getsource(mdl.resolve)
        for unusable in ("image", "tts", "audio"):
            assert unusable in source

    def test_the_key_never_reaches_a_log(self):
        assert mdl.redact("https://x/?key=SECRETVALUE", key="SECRETVALUE") == (
            "https://x/?key=<redacted>"
        )

    def test_ensure_env_clears_the_vertex_switches(self, monkeypatch):
        """Either one set routes to Vertex, which needs billing."""
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "something")
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        mdl.ensure_env()
        import os

        assert "GOOGLE_CLOUD_PROJECT" not in os.environ
        assert "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ


class TestAQuotaFailureIsNotAScore:
    def test_exhaustion_raises_rather_than_returning_a_failed_outcome(self):
        """Counting an unreachable scenario as a failed one would understate
        the agent by however many scenarios were left."""
        import inspect

        source = inspect.getsource(build.adk_runner)
        assert "raise QuotaExhausted" in source

    def test_a_partial_run_is_marked_partial(self):
        import inspect

        source = inspect.getsource(build.main)
        assert '"partial": len(attempted) < len(SUITE)' in source

    def test_each_scenario_is_driven_exactly_once(self):
        """Letting the scorer drive the agent would spend the quota twice."""
        import inspect

        source = inspect.getsource(build.main)
        assert "cache[scenario.id] = live(scenario)" in source
        assert "lambda scenario: cache[scenario.id]" in source


class TestTheStoredRun:
    @pytest.fixture(scope="class")
    def payload(self):
        path = build.ARTIFACT_DIR / "adk.json"
        if not path.exists():
            pytest.skip("day 16 build not run")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_records_which_model_produced_the_numbers(self, payload):
        """Model availability on a free key changes; the artifact must say
        which one answered rather than leaving a reader to guess."""
        assert payload["model"]
        assert payload["models_listed"] > 0

    def test_it_says_how_much_of_the_suite_ran(self, payload):
        assert payload["scenarios"] <= payload["of_suite"]
        if payload["scenarios"] < payload["of_suite"]:
            assert payload["partial"]
            assert payload["quota_note"]

    def test_the_other_two_bots_are_stored_beside_it(self, payload):
        for key in ("engine", "schema_only", "grounded"):
            assert payload[key], key

    def test_every_pass_rate_is_a_rate(self, payload):
        for key in ("adk", "engine", "schema_only", "grounded"):
            rate = payload[key]["pass_rate"]
            assert 0.0 <= rate <= 1.0, (key, rate)


class TestADKStaysOutOfTheServingPath:
    """The framework may not follow its own artifact into the hub.

    Every other day keeps its heavy dependency behind a lazy import so the
    serving image carries none of it; `google-adk` pulls in a large tree and
    is the easiest of them to leak, because day 16's tab sits next to the code
    that needs it. CI blocks the import inside the built image; this blocks it
    here, where the failure is one line instead of a red container step.
    """

    @pytest.fixture
    def blocked(self):
        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "google.adk" or name.startswith("google.adk."):
                    raise ImportError(f"blocked for this test: {name}")
                return None

        blocker = Block()
        sys.meta_path.insert(0, blocker)
        for name in [n for n in sys.modules if n.startswith("convo.projects.day16")]:
            del sys.modules[name]
        try:
            yield
        finally:
            sys.meta_path.remove(blocker)

    def test_the_tab_imports_without_the_framework(self, blocked):
        import importlib

        for module in ("results", "ui", "model", "build", ""):
            name = f"convo.projects.day16_adk_agent{'.' + module if module else ''}"
            importlib.import_module(name)

    def test_the_stored_run_loads_without_the_framework(self, blocked):
        import importlib

        results = importlib.import_module("convo.projects.day16_adk_agent.results")
        results.load.cache_clear()
        results.load()


class TestNotMeasuredIsNotZero:
    """The distinction day 6 built `scored_in` for, applied to a fourth bot.

    Day 5's runs predate the dimensions, so `pass_rate_in` returns 0.0 for all
    three of them. Rendering that as **0%** would put a scarlet zero under
    day 5's safety column and claim a measurement that was never made — the
    same class of mistake day 15 caught in day 12's latency.
    """

    @pytest.fixture(scope="class")
    def comparison(self):
        from convo.projects.day16_adk_agent.results import load

        found = load()
        if found is None:
            pytest.skip("day 16 build not run")
        return found

    def test_day_fives_dimensions_are_none_not_zero(self, comparison):
        from convo.core.evals import DIMENSIONS
        from convo.projects.day16_adk_agent.results import Scope

        for key in ("schema_only", "grounded"):
            bot = comparison.bot(key)
            if bot is None or bot.run(Scope.DAY5) is None:
                continue
            for dimension in DIMENSIONS:
                assert bot.dimension(Scope.DAY5, dimension) is None, (key, dimension)

    def test_day_fives_model_calls_are_none_not_zero(self, comparison):
        """Day 5's agent made thousands of calls; its stored run just does not
        say so. Reporting 0 would be a false claim about a model-driven bot."""
        from convo.projects.day16_adk_agent.results import Scope

        for key in ("schema_only", "grounded"):
            bot = comparison.bot(key)
            if bot is None or bot.run(Scope.DAY5) is None:
                continue
            assert bot.model_calls(Scope.DAY5) is None, key

    def test_the_state_machine_may_report_zero_calls(self, comparison):
        """The one run that can claim zero without measuring it: no model."""
        from convo.projects.day16_adk_agent.results import Scope

        assert comparison.engine.model_calls(Scope.FULL) == 0

    def test_adk_records_its_own_calls(self, comparison):
        from convo.projects.day16_adk_agent.results import Scope

        assert (comparison.adk.model_calls(Scope.FULL) or 0) > 0

    def test_adk_measures_all_three_dimensions(self, comparison):
        from convo.core.evals import DIMENSIONS
        from convo.projects.day16_adk_agent.results import Scope

        for dimension in DIMENSIONS:
            assert comparison.adk.dimension(Scope.FULL, dimension) is not None, dimension


class TestTheTwoScopesShareDenominators:
    @pytest.fixture(scope="class")
    def comparison(self):
        from convo.projects.day16_adk_agent.results import load

        found = load()
        if found is None:
            pytest.skip("day 16 build not run")
        return found

    def test_every_bot_in_a_scope_covers_exactly_that_scope(self, comparison):
        """A rate is only comparable if the denominators match exactly."""
        from convo.projects.day16_adk_agent.results import Scope

        for scope in (Scope.FULL, Scope.DAY5):
            expected = set(comparison.ids(scope))
            if not expected:
                continue
            for bot in comparison.ordered(scope):
                got = {c.case_id for c in bot.run(scope).cases}
                assert got == expected, (scope, bot.key, expected - got, got - expected)

    def test_day_fives_scope_is_inside_the_full_one(self, comparison):
        from convo.projects.day16_adk_agent.results import Scope

        if comparison.ids(Scope.DAY5):
            assert set(comparison.ids(Scope.DAY5)) <= set(comparison.ids(Scope.FULL))

    def test_day_five_is_absent_from_the_full_scope(self, comparison):
        """Its suite was smaller, so it cannot appear in the full-suite table."""
        from convo.projects.day16_adk_agent.results import Scope

        assert {b.key for b in comparison.ordered(Scope.FULL)} <= {"engine", "adk"}

    def test_the_false_confirmation_count_uses_day_sixes_assertion(self, comparison):
        """Day 5's rule cannot see a bot that booked when it should not have,
        which is the only false confirmation ADK actually commits."""
        from convo.projects.day16_adk_agent.results import FALSE_CONFIRMATION, Scope

        run = comparison.adk.run(Scope.FULL)
        explicit = sum(
            1
            for case in run.cases
            for a in case.assertions
            if a.name == FALSE_CONFIRMATION and not a.passed
        )
        assert comparison.adk.false_confirmations(Scope.FULL) == explicit

    def test_rescoring_never_touches_the_adk_run(self):
        """The ADK run costs a day's quota; the comparators are free. The
        rescore path must not be able to quietly overwrite the expensive half."""
        import inspect

        source = inspect.getsource(build._rescore_stored)
        assert 'payload["adk"] =' not in source
        for comparator in ("engine", "schema_only", "grounded"):
            assert f'payload["{comparator}"] =' in source, comparator
