"""Day 19: that the guarantee is real, the oracle is derived, and the scoring
separates a misroute from a booking written after a handoff.

Nothing here calls a model and nothing here needs a key. The tests that build
the agent tree need `google-adk` installed and skip without it — the same
property day 18's suite relies on, and the reason the topology audit can run
in a CI job that cannot run the bot.

The falsifiability tests matter more than the passing ones. An audit that only
ever sees a correct topology is not evidence of anything, so several tests
mutate the graph and require the audit to complain.
"""

from __future__ import annotations

import importlib
import json
import sys
from importlib.abc import MetaPathFinder

import pytest

from convo.core.evals import Assertion, CaseResult, EvalRun
from convo.projects.day06_eval_harness.suite import SUITE
from convo.projects.day18_eval_gate.trajectory import expected_for
from convo.projects.day19_multi_agent import oracle as orc
from convo.projects.day19_multi_agent import score as sc
from convo.projects.day19_multi_agent.agents import (
    BOOK_TOOL,
    BOOKER,
    COLLECTOR,
    HANDOFF,
    POLICY,
    POLICY_TOOL,
    ROUTER,
    WRAPPER_FOR,
)


def _has_adk() -> bool:
    try:
        import google.adk  # noqa: F401
    except Exception:
        return False
    return True


needs_adk = pytest.mark.skipif(not _has_adk(), reason="google-adk is not installed")


# ----------------------------------------------------------------------------
# The guarantee, as a graph.
# ----------------------------------------------------------------------------


@needs_adk
class TestTheTopologyIsAProof:
    def test_the_audit_is_silent_on_the_real_topology(self):
        from convo.projects.day19_multi_agent import scope

        assert scope.audit(scope.derive()) == []

    def test_it_needs_no_api_key(self, monkeypatch):
        """CI has no key, and the audit is the half of day 19 that runs there.

        Asserted rather than assumed: day 18 cost three red builds to a
        dependency property I stated four times before running it once.
        """
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from convo.projects.day19_multi_agent import scope

        assert scope.derive().agents

    def test_the_booking_tool_has_exactly_one_holder(self):
        from convo.projects.day19_multi_agent import scope

        assert scope.derive().holders_of(BOOK_TOOL) == (BOOKER,)

    def test_the_handoff_desk_holds_nothing(self):
        from convo.projects.day19_multi_agent import scope

        assert scope.derive().tools[HANDOFF] == ()

    def test_control_cannot_leave_the_handoff_desk(self):
        from convo.projects.day19_multi_agent import scope

        assert scope.derive().reachable_from(HANDOFF) == frozenset({HANDOFF})

    def test_no_tool_is_reachable_once_handed_off(self):
        """The claim the whole day rests on."""
        from convo.projects.day19_multi_agent import scope

        assert scope.derive().tools_reachable_from(HANDOFF) == frozenset()

    def test_both_tools_are_still_reachable_from_the_router(self):
        """Otherwise the guarantee holds because the feature is missing."""
        from convo.projects.day19_multi_agent import scope

        reachable = scope.derive().tools_reachable_from(ROUTER)
        assert BOOK_TOOL in reachable
        assert POLICY_TOOL in reachable

    def test_the_wrapper_name_is_mapped_back_to_the_executed_tool(self):
        """The model is offered `book_a_table`; `book_table` is what runs."""
        from convo.projects.day19_multi_agent import scope

        graph = scope.derive()
        assert WRAPPER_FOR[BOOK_TOOL] == "book_a_table"
        assert BOOK_TOOL in graph.tools[BOOKER]
        assert "book_a_table" not in graph.tools[BOOKER]


@needs_adk
class TestTheAuditCanFail:
    """A check that cannot fail is decoration. These make it fail."""

    def test_it_catches_a_handoff_desk_that_can_transfer_out(self):
        from convo.projects.day19_multi_agent import scope

        graph = scope.derive()
        leaky = scope.Topology(
            tools=graph.tools,
            edges={**graph.edges, HANDOFF: (ROUTER,)},
        )
        complaints = scope.audit(leaky)
        assert any("control can leave" in c for c in complaints)
        assert any(BOOK_TOOL in c for c in complaints)

    def test_it_catches_a_second_holder_of_the_booking_tool(self):
        from convo.projects.day19_multi_agent import scope

        graph = scope.derive()
        shared = scope.Topology(
            tools={**graph.tools, COLLECTOR: (BOOK_TOOL,)},
            edges=graph.edges,
        )
        assert any(BOOK_TOOL in c and BOOKER in c for c in scope.audit(shared))

    def test_it_catches_a_bot_that_can_never_book(self):
        from convo.projects.day19_multi_agent import scope

        graph = scope.derive()
        toothless = scope.Topology(
            tools={**graph.tools, BOOKER: ()},
            edges=graph.edges,
        )
        assert any("not reachable from" in c for c in scope.audit(toothless))

    def test_it_catches_tools_left_on_the_handoff_desk(self):
        from convo.projects.day19_multi_agent import scope

        graph = scope.derive()
        armed = scope.Topology(
            tools={**graph.tools, HANDOFF: (BOOK_TOOL,)},
            edges=graph.edges,
        )
        assert any(HANDOFF in c for c in scope.audit(armed))


# ----------------------------------------------------------------------------
# The oracle.
# ----------------------------------------------------------------------------


class TestTheRouteIsDerived:
    def test_it_agrees_with_every_scenario_label(self):
        for scenario in SUITE:
            route = orc.route_for(scenario)
            assert orc.disagrees_with_suite(scenario, route) == "", scenario.id

    def test_no_turn_forces_a_choice_between_two_desks(self):
        """The precedence in `oracle` is only honest while this holds."""
        for scenario in SUITE:
            assert orc.arbitration_needed(scenario) == "", scenario.id

    def test_it_matches_day_18s_independently_derived_tools(self):
        """Two oracles off the same engine. A drift would look like a finding."""
        for scenario in SUITE:
            route = orc.route_for(scenario)
            assert orc.expected_tools(route) == expected_for(scenario).per_turn, scenario.id

    def test_the_handoff_desk_is_sticky(self):
        """Day 4 makes HANDOFF terminal, so no later turn may leave it."""
        for scenario in SUITE:
            desks = orc.route_for(scenario).per_turn
            if HANDOFF in desks:
                first = desks.index(HANDOFF)
                assert set(desks[first:]) == {HANDOFF}, scenario.id

    def test_handoff_turn_is_the_first_one(self):
        for scenario in SUITE:
            route = orc.route_for(scenario)
            if HANDOFF in route.per_turn:
                assert route.handoff_turn == route.per_turn.index(HANDOFF)
            else:
                assert route.handoff_turn is None

    def test_the_guarantee_covers_all_but_one_must_not_book_scenario(self):
        """Not *all* of them, which is what I assumed before running it.

        Six scenarios must not book. Five reach a handoff and are what the
        topology protects. `refuses_a_rule_override` is refused inside the
        booking tool instead — the engine stays in `collect` and the slot
        validator rejects a party of 50 — so day 19's guarantee does nothing
        for it, and the write-up has to say which mechanism covers what.
        """
        handoff = {s.id for s in SUITE if orc.route_for(s).handoff_turn is not None}
        must_not = {s.id for s in SUITE if not s.expect_booked}
        assert handoff < must_not
        assert must_not - handoff == {"refuses_a_rule_override"}

    def test_the_uncovered_scenario_never_leaves_the_collector(self):
        route = orc.route_for(next(s for s in SUITE if s.id == "refuses_a_rule_override"))
        assert set(route.per_turn) == {COLLECTOR}
        assert not route.books

    def test_a_booking_scenario_reaches_the_booking_desk_exactly_once(self):
        for scenario in SUITE:
            route = orc.route_for(scenario)
            if route.books:
                assert route.per_turn.count(BOOKER) == 1, scenario.id
                assert route.per_turn[route.booked_on] == BOOKER

    def test_a_digression_routes_to_the_policy_desk(self):
        found = False
        for scenario in SUITE:
            route = orc.route_for(scenario)
            for index, digressed in enumerate(route.digressed):
                if digressed and route.per_turn[index] != HANDOFF:
                    assert route.per_turn[index] == POLICY, scenario.id
                    found = True
        assert found, "no scenario digresses, so this test proves nothing"


# ----------------------------------------------------------------------------
# Scoring.
# ----------------------------------------------------------------------------


def _turn(authors, tools=(), by_agent=None):
    return sc.Turn(
        authors=tuple(authors),
        tools=tuple(tools),
        by_agent=tuple(by_agent or [(BOOKER, t) for t in tools]),
    )


def _route(per_turn, booked_on=None):
    return orc.Route(
        scenario_id="synthetic",
        per_turn=tuple(per_turn),
        digressed=tuple(False for _ in per_turn),
        booked_on=booked_on,
    )


class TestScoringTheRouter:
    def test_a_clean_route_scores_everything(self):
        route = _route([COLLECTOR, BOOKER], booked_on=1)
        turns = (_turn([ROUTER, COLLECTOR]), _turn([ROUTER, BOOKER], [BOOK_TOOL]))
        verdict = sc.score_route(route, turns)
        assert verdict.misroutes == ()
        assert verdict.routing_accuracy == 1.0
        assert verdict.clean

    def test_the_expected_desk_only_has_to_be_among_the_authors(self):
        """ADK's root agent authors the transfer, so it is on nearly every turn."""
        route = _route([COLLECTOR])
        verdict = sc.score_route(route, (_turn([ROUTER, COLLECTOR]),))
        assert verdict.misroutes == ()

    def test_a_misroute_is_caught(self):
        route = _route([HANDOFF])
        verdict = sc.score_route(route, (_turn([ROUTER, COLLECTOR]),))
        assert verdict.misroutes == (0,)
        assert verdict.routing_accuracy == 0.0

    def test_a_booking_after_the_handoff_is_late(self):
        """Day 16's reproducing defect, which the topology exists to prevent."""
        route = _route([COLLECTOR, HANDOFF])
        turns = (_turn([ROUTER, COLLECTOR]), _turn([ROUTER, BOOKER], [BOOK_TOOL]))
        verdict = sc.score_route(route, turns)
        assert verdict.late_booking
        assert verdict.late_booking_turn == 1

    def test_a_booking_before_the_handoff_is_not_late(self):
        route = _route([BOOKER, HANDOFF], booked_on=0)
        turns = (_turn([ROUTER, BOOKER], [BOOK_TOOL]), _turn([ROUTER, HANDOFF]))
        assert not sc.score_route(route, turns).late_booking

    def test_a_scenario_with_no_handoff_can_never_book_late(self):
        route = _route([COLLECTOR, BOOKER], booked_on=1)
        turns = (_turn([ROUTER, COLLECTOR]), _turn([ROUTER, BOOKER], [BOOK_TOOL]))
        assert sc.score_route(route, turns).late_booking_turn is None

    def test_a_scope_breach_is_attributed(self):
        """Should be impossible. Checked at runtime because a graph proof is not
        a proof about a framework's behaviour."""
        route = _route([COLLECTOR])
        turns = (_turn([ROUTER, COLLECTOR], [BOOK_TOOL], [(COLLECTOR, BOOK_TOOL)]),)
        verdict = sc.score_route(route, turns)
        assert verdict.scope_breaches == ((COLLECTOR, BOOK_TOOL),)
        assert not verdict.clean

    def test_the_booking_desk_running_the_tool_is_no_breach(self):
        route = _route([BOOKER], booked_on=0)
        turns = (_turn([ROUTER, BOOKER], [BOOK_TOOL], [(BOOKER, BOOK_TOOL)]),)
        assert sc.score_route(route, turns).scope_breaches == ()

    def test_nothing_scored_is_none_not_zero(self):
        """Day 5's lesson: an unmeasured dimension rendered as 0.0 is a claim
        nobody made."""
        verdict = sc.score_route(_route([]), ())
        assert verdict.routing_accuracy is None

    def test_a_short_run_scores_only_the_turns_it_has(self):
        route = _route([COLLECTOR, BOOKER], booked_on=1)
        verdict = sc.score_route(route, (_turn([ROUTER, COLLECTOR]),))
        assert verdict.scored_turns == 1


class TestTheSummary:
    def _summary(self):
        clean = sc.score_route(
            _route([COLLECTOR, BOOKER], booked_on=1),
            (_turn([ROUTER, COLLECTOR]), _turn([ROUTER, BOOKER], [BOOK_TOOL])),
        )
        leaked = sc.score_route(
            _route([COLLECTOR, HANDOFF]),
            (_turn([ROUTER, COLLECTOR]), _turn([ROUTER, BOOKER], [BOOK_TOOL])),
        )
        held = sc.score_route(
            _route([COLLECTOR, HANDOFF]),
            (_turn([ROUTER, COLLECTOR]), _turn([ROUTER, HANDOFF])),
        )
        return sc.Summary(verdicts=(clean, leaked, held))

    def test_accuracy_is_over_turns_not_scenarios(self):
        """Six turns, one misroute: 5/6, not an average of per-scenario rates."""
        summary = self._summary()
        assert summary.turns == 6
        assert summary.routing_accuracy == round(5 / 6, 4)

    def test_late_bookings_are_listed(self):
        assert len(self._summary().late_bookings) == 1

    def test_handoff_scenarios_are_the_ones_with_a_handoff_expected(self):
        assert len(self._summary().handoff_scenarios) == 2

    def test_handoff_held_counts_only_the_clean_ones(self):
        assert self._summary().handoff_held == 1

    def test_it_serialises(self):
        payload = json.loads(json.dumps(self._summary().as_dict()))
        assert payload["handoff_held"] == 1
        assert len(payload["verdicts"]) == 3


# ----------------------------------------------------------------------------
# The artifact, which does not exist yet.
# ----------------------------------------------------------------------------


class TestResultsWithNoRun:
    def test_load_returns_none_rather_than_raising(self):
        from convo.projects.day19_multi_agent.results import ARTIFACT_FILE, load

        load.cache_clear()
        if not ARTIFACT_FILE.exists():
            assert load() is None

    def test_the_experiment_describes_itself_without_an_artifact(self):
        """The topology and the five scenarios are free and deterministic, which
        is what lets the tab be honest before the run is paid for."""
        from convo.projects.day19_multi_agent.results import handoff_scenarios

        assert len(handoff_scenarios()) == 5

    def test_the_targets_were_named_in_advance(self):
        from convo.projects.day19_multi_agent.results import PENDING_TARGETS, handoff_scenarios

        assert set(PENDING_TARGETS) <= set(handoff_scenarios())


def _results(verdicts: dict[str, bool], day16: dict[str, str]):
    from convo.projects.day19_multi_agent.results import Results

    cases = [
        CaseResult.from_assertions(
            scenario,
            [Assertion(name="booked", passed=passed, dimension="safety")],
        )
        for scenario, passed in verdicts.items()
    ]
    run = EvalRun(suite="t", run_id="t", started_at="t", cases=cases)
    return Results(
        model="none",
        baseline_model="none",
        model_drift=False,
        scenarios=len(cases),
        of_suite=32,
        partial=False,
        quota_note="",
        unavailable=(),
        topology={},
        run=run,
        engine=run,
        rows=(),
        routing_accuracy=None,
        handoff_held=0,
        transfers={},
        day16_stored=day16,
        day16_corrected=day16,
        day16_stale=False,
    )


class TestComparingAgainstDay16:
    def test_it_reports_what_the_router_fixed(self):
        results = _results({"a": True}, {"a": "fail"})
        assert results.fixed_what_day16_broke() == ("a",)
        assert results.broke_what_day16_held() == ()

    def test_it_reports_what_the_router_broke_with_equal_prominence(self):
        results = _results({"a": False}, {"a": "pass"})
        assert results.broke_what_day16_held() == ("a",)
        assert results.fixed_what_day16_broke() == ()

    def test_a_scenario_neither_ran_is_claimed_either_way(self):
        results = _results({}, {"a": "fail"})
        assert results.fixed_what_day16_broke() == ()
        assert results.broke_what_day16_held() == ()


# ----------------------------------------------------------------------------
# The preflight, and the baseline it refuses to flatter.
# ----------------------------------------------------------------------------


@needs_adk
class TestThePreflight:
    def test_it_passes_on_the_committed_state(self):
        from convo.projects.day19_multi_agent.build import preflight

        assert preflight(verbose=False) == []

    def test_it_runs_with_no_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from convo.projects.day19_multi_agent.build import preflight

        assert preflight(verbose=False) == []


class TestTheBaselineIsNotFlattered:
    def test_the_known_false_positive_is_corrected(self):
        """Day 16's artifact records `never_books_without_details` as a failure
        and it was not one. Taking the stored verdict would credit the router
        with fixing a conversation the single agent never got wrong."""
        from convo.projects.day19_multi_agent.build import (
            DAY16_CORRECTIONS,
            day16_baseline,
            day16_corrected,
            day16_stale,
        )

        if not day16_stale():
            pytest.skip("day 16 has been re-run; the correction no longer applies")
        scenario = "never_books_without_details"
        assert day16_baseline().get(scenario) == "fail"
        assert day16_corrected().get(scenario) == "pass"
        assert DAY16_CORRECTIONS[scenario] == "pass"

    def test_the_two_targets_really_did_fail(self):
        """If day 16 passed them, day 19 has nothing to test and the write-up
        must say so instead of reporting an improvement."""
        from convo.projects.day19_multi_agent.build import day16_corrected
        from convo.projects.day19_multi_agent.results import PENDING_TARGETS

        corrected = day16_corrected()
        if not corrected:
            pytest.skip("day 16 has no artifact")
        for target in PENDING_TARGETS:
            assert corrected[target] == "fail", target

    def test_the_correction_disappears_once_day_16_is_rerun(self, monkeypatch):
        from convo.projects.day19_multi_agent import build

        monkeypatch.setattr(build, "day16_stale", lambda: False)
        monkeypatch.setattr(
            build, "day16_baseline", lambda: {"never_books_without_details": "fail"}
        )
        assert build.day16_corrected() == {"never_books_without_details": "fail"}


class TestTheRetryTaxonomyIsShared:
    def test_day_19_does_not_redefine_day_16s_error_classes(self):
        """Two copies would drift, and each rule in it cost a day's quota."""
        from convo.projects.day16_adk_agent import build as d16
        from convo.projects.day19_multi_agent import build as d19

        assert d19.QuotaExhausted is d16.QuotaExhausted
        assert d19.ScenarioUnavailable is d16.ScenarioUnavailable
        assert d19.BACKOFF_S is d16.BACKOFF_S


class TestAThinRunIsNotAResult:
    """The cross-model run reached one scenario. A rate over it would be the
    most misleading honest number in the repository.

    `gemini-3.5-flash` allows 20 requests a day against `flash-lite`'s 500, so
    the comparison was not imprecise, it was unreachable: the five handoff
    conversations sit at suite positions 12 to 20.
    """

    def test_the_real_cross_model_run_is_refused(self):
        from convo.projects.day19_multi_agent.results import load, load_others

        load.cache_clear()
        load_others.cache_clear()
        controlled = load()
        others = load_others()
        if controlled is None or not others:
            pytest.skip("no cross-model run on disk")
        for other in others:
            if other.scenarios < 8:
                assert controlled.too_thin_to_compare(other)

    def test_too_few_shared_scenarios_is_refused(self):
        from convo.projects.day19_multi_agent.results import MIN_SHARED_SCENARIOS

        controlled = _results({f"s{i}": True for i in range(20)}, {})
        thin = _results({"s0": True}, {})
        reason = controlled.too_thin_to_compare(thin)
        assert str(MIN_SHARED_SCENARIOS) in reason

    def test_missing_handoff_coverage_is_refused(self):
        """Enough scenarios, none of them the ones the day is about."""
        shared = {f"s{i}": True for i in range(12)}
        controlled = _results(shared, {})
        other = _results(shared, {})
        assert "structural claim" in controlled.too_thin_to_compare(other)

    def test_a_comparable_pair_is_allowed(self):
        from convo.projects.day19_multi_agent.results import handoff_scenarios

        ids = {name: True for name in list(handoff_scenarios())[:2]}
        ids.update({f"s{i}": True for i in range(10)})
        controlled = _results(ids, {})
        other = _results(ids, {})
        assert controlled.too_thin_to_compare(other) == ""

    def test_rates_are_computed_on_the_shared_set_only(self):
        """Comparing a 32-scenario rate against a 24-scenario one is day 5's
        denominator mistake."""
        controlled = _results({"a": True, "b": True, "c": False}, {})
        rates = controlled.rates_on(frozenset({"a", "b"}))
        assert rates["pass_rate"] == 1.0

    def test_an_unclaimed_dimension_is_none_not_zero(self):
        controlled = _results({"a": True}, {})
        assert controlled.rates_on(frozenset({"a"}))["task"] is None

    def test_an_empty_overlap_reports_nothing_rather_than_zero(self):
        controlled = _results({"a": True}, {})
        assert controlled.rates_on(frozenset())["pass_rate"] is None


class TestTheQuotaNoteKeepsTheQuota:
    """Day 19's cross-model artifact recorded a quota note with no quota in it.

    Google puts a mitigation link and boilerplate first, so taking the leading
    300 characters reliably cut the one line that says which allowance was hit
    and how big it is. The limit it hid was 20 per day.
    """

    def test_it_keeps_the_limit_line(self):
        from convo.projects.day16_adk_agent.build import quota_detail

        note = quota_detail(
            RuntimeError(
                "On how to mitigate this issue, see https://example/docs\n\n"
                "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
                "generate_content_free_tier_requests, limit: 20, model: "
                "gemini-3.5-flash, description: per day limit exceeded."
            )
        )
        assert "limit: 20" in note
        assert "mitigate" not in note

    def test_it_falls_back_to_the_message_when_there_is_no_quota_line(self):
        from convo.projects.day16_adk_agent.build import quota_detail

        assert "something else broke" in quota_detail(RuntimeError("something else broke"))

    def test_it_stays_within_the_stored_length(self):
        from convo.projects.day16_adk_agent.build import quota_detail

        assert len(quota_detail(RuntimeError("x" * 5000))) <= 300


class TestACrossModelRunGetsItsOwnFile:
    """The pin stops a drifted run being mistaken for the controlled one.
    This stops it replacing it.

    Writing a cross-model run to `multi_agent.json` would overwrite the result
    day 19's published figures rest on, and the overwrite would look like an
    update: the comparison against day 16 would still render, still be labelled
    "router against one agent", and silently be measuring the model.
    """

    def test_the_baseline_model_keeps_the_canonical_path(self):
        from convo.projects.day19_multi_agent.build import (
            ARTIFACT_FILE,
            artifact_for,
            baseline_model,
        )

        want = baseline_model()
        if not want:
            pytest.skip("day 16 has no artifact")
        assert artifact_for(want) == ARTIFACT_FILE

    def test_another_model_gets_its_own_path(self):
        from convo.projects.day19_multi_agent.build import ARTIFACT_FILE, artifact_for

        other = artifact_for("gemini-9.9-imaginary")
        assert other != ARTIFACT_FILE
        assert "gemini-9.9-imaginary" in other.name

    def test_load_returns_only_the_controlled_run(self):
        from convo.projects.day19_multi_agent.results import load

        load.cache_clear()
        found = load()
        if found is None:
            pytest.skip("day 19 has not been run")
        assert found.model_drift is False
        assert found.model == found.baseline_model

    def test_other_runs_are_read_separately_and_know_they_drifted(self):
        from convo.projects.day19_multi_agent.results import load_others

        load_others.cache_clear()
        for other in load_others():
            assert other.model_drift is True
            assert other.model != other.baseline_model

    def test_rescoring_covers_every_run_not_just_the_controlled_one(self):
        """Otherwise a cross-model figure outlives the scorer that made it."""
        from convo.projects.day19_multi_agent.build import ARTIFACT_FILE, stored_artifacts

        paths = stored_artifacts()
        if not paths:
            pytest.skip("day 19 has not been run")
        assert ARTIFACT_FILE in paths
        from convo.projects.day19_multi_agent.results import ARTIFACT_DIR

        assert len(paths) == len(list(ARTIFACT_DIR.glob("multi_agent*.json")))


def _artifact_without_drift_fields():
    """A copy of day 19's artifact as a pre-pin build would have written it.

    Written to a temp file rather than mutating the committed one: the point of
    the test is that an old artifact still answers, not that artifacts may be
    edited by hand.
    """
    import tempfile
    from pathlib import Path

    from convo.projects.day19_multi_agent.results import ARTIFACT_FILE

    payload = json.loads(ARTIFACT_FILE.read_text(encoding="utf-8"))
    payload.pop("baseline_model", None)
    payload.pop("model_drift", None)
    payload["model"] = "gemini-something-else"
    out = Path(tempfile.mkdtemp()) / "multi_agent.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


class TestTheModelIsAControlledVariable:
    """`resolve()` falls back; a comparison must not.

    Caught before it did any damage, by probing the quota before running:
    `flash-lite`'s daily allowance was spent, `resolve()` had already moved on
    to `flash`, and a full 32-scenario run would have written an artifact that
    compared cleanly against day 16 and measured the model instead of the
    topology.
    """

    def test_the_baseline_model_comes_from_day_16s_artifact(self):
        """Read, not named here, so the two cannot drift apart."""
        from convo.projects.day19_multi_agent.build import baseline_model

        found = baseline_model()
        if not found:
            pytest.skip("day 16 has no artifact")
        assert found.startswith("gemini")

    def test_it_refuses_rather_than_substituting(self, monkeypatch):
        from convo.projects.day16_adk_agent.model import ModelUnavailable
        from convo.projects.day19_multi_agent import build

        monkeypatch.setattr(build, "baseline_model", lambda: "gemini-pinned")

        def gone(name):
            raise ModelUnavailable(f"{name} would not serve (daily quota exhausted)")

        monkeypatch.setattr(build, "require", gone)
        monkeypatch.setattr(build, "resolve", lambda: pytest.fail("resolve() must not be reached"))
        with pytest.raises(SystemExit) as caught:
            build.pinned(allow_drift=False)
        assert "gemini-pinned" in str(caught.value)
        assert "--allow-model-drift" in str(caught.value)

    def test_drift_is_opt_in_and_reported(self, monkeypatch):
        from convo.projects.day16_adk_agent.model import Resolved
        from convo.projects.day19_multi_agent import build

        monkeypatch.setattr(build, "baseline_model", lambda: "gemini-pinned")
        monkeypatch.setattr(build, "resolve", lambda: Resolved("gemini-other", 1, ()))
        resolved, drift = build.pinned(allow_drift=True)
        assert resolved.model == "gemini-other"
        assert drift is True

    def test_no_drift_when_the_available_model_is_the_baseline(self, monkeypatch):
        from convo.projects.day16_adk_agent.model import Resolved
        from convo.projects.day19_multi_agent import build

        monkeypatch.setattr(build, "baseline_model", lambda: "gemini-pinned")
        monkeypatch.setattr(build, "resolve", lambda: Resolved("gemini-pinned", 1, ()))
        _, drift = build.pinned(allow_drift=True)
        assert drift is False

    def test_a_first_run_has_nothing_to_pin_to(self, monkeypatch):
        from convo.projects.day16_adk_agent.model import Resolved
        from convo.projects.day19_multi_agent import build

        monkeypatch.setattr(build, "baseline_model", lambda: "")
        monkeypatch.setattr(build, "resolve", lambda: Resolved("gemini-whatever", 1, ()))
        resolved, drift = build.pinned(allow_drift=False)
        assert resolved.model == "gemini-whatever"
        assert drift is False

    def test_the_loaded_run_knows_whether_it_is_cross_model(self):
        """So a tab can state it without opening day 16's artifact.

        Derived for the run on disk, which predates the pin, and read straight
        off the payload for anything built after it. Both paths have to answer.
        """
        from convo.projects.day19_multi_agent.results import load

        load.cache_clear()
        found = load()
        if found is None:
            pytest.skip("day 19 has not been run")
        assert found.baseline_model
        assert found.model == found.baseline_model
        assert found.model_drift is False

    def test_drift_is_derived_when_the_artifact_predates_the_field(self, monkeypatch):
        from convo.projects.day19_multi_agent import results

        monkeypatch.setattr(results, "ARTIFACT_FILE", _artifact_without_drift_fields())
        results.load.cache_clear()
        found = results.load()
        assert found is not None
        assert found.model_drift is True
        results.load.cache_clear()


class TestTheProofIsReadableWithoutTheFramework:
    """The failure that cost a red Pages build, turned into tests.

    The reachability claim is derived from the real agent objects, which
    imports `google-adk`. The static site and the container are built in jobs
    that deliberately do not install it — day 18's CI even has a leak guard
    naming `google.adk`. So the graph is committed, and everything downstream
    reads the artifact.
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
        # A meta-path hook is never consulted for a module already imported, and
        # earlier tests in this file import ADK. So the cached entries come out
        # too, and go back afterwards — without this the blocker silently does
        # nothing and every test under it passes for the wrong reason.
        cached = {
            n: m for n, m in sys.modules.items() if n == "google.adk" or n.startswith("google.adk.")
        }
        for name in cached:
            del sys.modules[name]
        for name in [n for n in sys.modules if n.startswith("convo.projects.day19")]:
            del sys.modules[name]
        try:
            yield
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(cached)

    def test_every_day19_module_imports(self, blocked):
        for module in ("agents", "scope", "oracle", "score", "results", "build", "ui", ""):
            importlib.import_module(
                f"convo.projects.day19_multi_agent{'.' + module if module else ''}"
            )

    def test_the_committed_graph_loads(self, blocked):
        scope = importlib.import_module("convo.projects.day19_multi_agent.scope")
        assert scope.load() is not None

    def test_the_audit_passes_on_the_committed_graph(self, blocked):
        """The proof has to be checkable where the framework is absent, or it
        is not a gate — it is a thing that happens to run on my laptop."""
        scope = importlib.import_module("convo.projects.day19_multi_agent.scope")
        assert scope.audit(scope.load()) == []

    def test_deriving_it_is_what_needs_the_framework(self, blocked):
        scope = importlib.import_module("convo.projects.day19_multi_agent.scope")
        with pytest.raises(ImportError):
            scope.derive()

    def test_the_site_page_builds(self, blocked):
        """The exact call that took Pages down."""
        data = importlib.import_module("convo.site.data")
        summary, stats, tables, caveat = data._day19()
        assert summary and tables and caveat

    def test_the_tab_builds(self, blocked):
        import gradio as gr

        ui = importlib.import_module("convo.projects.day19_multi_agent.ui")
        with gr.Blocks():
            ui.build()


@needs_adk
class TestTheCommittedGraphCannotDrift:
    def test_the_code_still_derives_what_was_committed(self):
        """A committed proof nothing re-derives is just a file. Day 18's
        reproducibility gate, applied to a topology instead of a score."""
        from convo.projects.day19_multi_agent import scope

        assert scope.reproduces() == []

    def test_drift_is_reported(self, monkeypatch):
        from convo.projects.day19_multi_agent import scope

        real = scope.derive()
        monkeypatch.setattr(
            scope,
            "derive",
            lambda *a, **k: scope.Topology(
                tools={**real.tools, HANDOFF: (BOOK_TOOL,)}, edges=real.edges
            ),
        )
        assert any("tools differ" in c for c in scope.reproduces())


class TestTheTabRenders:
    def test_it_builds_with_no_artifact(self):
        import gradio as gr

        from convo.projects.day19_multi_agent.ui import build

        with gr.Blocks():
            build()
