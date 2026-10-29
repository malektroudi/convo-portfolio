"""Day 21: the close-out, and the audit of what the front pages published.

The day's claim is that a portfolio's headline numbers must be recomputed from
the artifacts they describe, because the ones that were not had drifted. A test
file for that day has an obvious trap: it can assert the drift it already
found, which proves nothing a month from now.

So these tests are about the machinery instead. That the audit cannot flatter
itself by checking a literal against itself. That a resolver genuinely reads
its artifact rather than returning a constant that happens to match. That the
two front ends draw from one list, so they cannot disagree about what the best
finding is — which is what they were doing. And that a summary which counts
its own file does not settle, which is a real fixpoint and the reason day 21
leaves itself out of its own inventory.

The one figure asserted outright is the lead card's kind. Day 2's 0.092 has no
artifact, and a future rebuild of day 2 must not be able to quietly promote it
to a measurement.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from convo.core.registry import projects
from convo.projects.day21_capstone import build, claims, ledger
from convo.projects.day21_capstone.results import load


@pytest.fixture(scope="module")
def fresh() -> dict:
    return build.payload()


@pytest.fixture(scope="module")
def stored():
    found = load()
    if found is None:
        pytest.skip("day 21's artifact has not been built")
    return found


class TestTheAuditCannotFlatterItself:
    """A `TESTIMONY` claim carries the literal that was typed in.

    If `audit()` compared that literal against itself, the one number in the
    repository nothing can check would be reported as holding — the exact
    failure the audit exists to catch, dressed as a pass.
    """

    def test_a_testimony_claim_derives_nothing(self):
        claim = claims.by_key("day02-max-confidence")
        assert claim.kind == claims.TESTIMONY
        assert claim.derived() is None

    def test_but_it_still_has_a_figure_to_show(self):
        claim = claims.by_key("day02-max-confidence")
        assert claim.value() == "0.092"

    def test_so_its_verdict_is_never_holds(self):
        rows = {row.key: row for row in claims.audit()}
        assert rows["day02-max-confidence"].verdict == "no artifact"
        assert not rows["day02-max-confidence"].holds

    def test_no_testimony_claim_anywhere_reports_a_verdict_of_holds(self):
        for row in claims.audit():
            if row.kind == claims.TESTIMONY:
                assert row.verdict == "no artifact", row.key

    def test_every_audited_figure_names_a_real_claim(self):
        keys = {c.key for c in claims.CLAIMS}
        for was in claims.WAS_PUBLISHED:
            assert was.key in keys, was.key

    def test_the_audit_covers_every_published_figure(self):
        assert len(claims.audit()) == len(claims.WAS_PUBLISHED)

    def test_the_tally_adds_up(self):
        rows = claims.audit()
        assert sum(claims.tally(rows).values()) == len(rows)


class TestAResolverReallyReadsItsArtifact:
    """The weakness of a derived figure is a resolver that is not derived.

    A function returning a hard-coded string would pass every comparison in
    this file. Removing the artifact underneath it is the only check that
    distinguishes the two, so each one is asked to produce `None` when its
    `load` returns nothing.
    """

    CASES = (
        ("day02-committed-run", "convo.projects.day02_intent_nlu.results"),
        ("day03-spread", "convo.projects.day03_embed_router.results"),
        ("day05-false-confirmations", "convo.projects.day05_tool_agent.results"),
        ("day12-token-to-sentence", "convo.projects.day12_voice.results"),
        ("day15-containment", "convo.projects.day15_analytics.results"),
        ("day18-agreement", "convo.projects.day18_eval_gate.results"),
        ("day20-clean", "convo.projects.day20_a2a_handoff.results"),
    )

    @pytest.mark.parametrize(("key", "module"), CASES)
    def test_it_returns_nothing_when_the_artifact_is_gone(self, key, module, monkeypatch):
        import importlib

        monkeypatch.setattr(importlib.import_module(module), "load", lambda: None)
        assert claims.by_key(key).derived() is None

    @pytest.mark.parametrize(("key", "module"), CASES)
    def test_and_something_when_it_is_there(self, key, module):
        assert claims.by_key(key).derived(), key

    def test_day19_reads_the_committed_topology_not_the_framework(self, monkeypatch):
        """It must resolve where ADK is not installed, which is most places."""
        from convo.projects.day19_multi_agent import scope

        monkeypatch.setattr(scope, "load", lambda: None)
        assert claims.by_key("day19-reachable").derived() is None


class TestTheClaimsAreWellFormed:
    def test_keys_are_unique(self):
        keys = [c.key for c in claims.CLAIMS]
        assert len(keys) == len(set(keys))

    def test_every_claim_names_a_day_in_the_plan(self):
        days = {p.day for p in projects()}
        for claim in claims.CLAIMS:
            assert claim.day in days, claim.key

    def test_every_kind_is_one_of_the_four(self):
        for claim in claims.CLAIMS:
            assert claim.kind in claims.KINDS, claim.key

    def test_derived_and_pinned_claims_have_a_resolver(self):
        for claim in claims.CLAIMS:
            if claim.kind in (claims.DERIVED, claims.PINNED):
                assert claim.resolve is not None, claim.key

    def test_testimony_claims_do_not(self):
        for claim in claims.CLAIMS:
            if claim.kind == claims.TESTIMONY:
                assert claim.resolve is None, claim.key
                assert claim.figure, claim.key

    def test_every_kind_has_a_one_line_meaning(self):
        for kind in claims.KINDS:
            assert claims.MEANS[kind]

    def test_several_findings_lead_and_they_span_the_weeks(self):
        leads = claims.leads()
        assert len(leads) >= 3
        weeks = {p.week for p in projects() if p.day in {c.day for c in leads}}
        assert weeks == {1, 2, 3}

    def test_the_lead_cards_all_have_something_to_show(self):
        for claim in claims.leads():
            assert claim.value(), claim.key


class TestTheTwoFrontEndsCannotDisagree:
    """They did. The hub led on days 3, 5 and 2; the site on 2, 5 and 12.

    Each had its own hand-written tuple, so the portfolio had two answers to
    the question of what it had found. The `lead` flag now lives with the
    claim, which makes the disagreement impossible rather than unlikely.
    """

    def test_they_show_the_same_findings(self):
        from convo.site.build import findings

        site = {(f.day, f.headline) for f in findings()}
        hub = {(c.day, c.headline) for c in claims.leads()}
        assert site == hub

    def test_the_site_carries_the_kind_through(self):
        from convo.site.build import findings

        for card in findings():
            assert card.kind in claims.KINDS, card.headline
            assert card.means

    def test_neither_keeps_a_findings_list_of_its_own(self):
        """The tuples that drifted are gone, not merely agreeing today.

        `tests/test_site.py` checks that neither module contains a
        metric-shaped literal any more. This checks the structural half: the
        data they render comes from here.
        """
        from convo.hub import app
        from convo.site import build as site_build

        assert not hasattr(site_build, "FINDINGS")
        assert not hasattr(app, "FINDINGS")


class TestASummaryCannotCountItself:
    """Writing the artifact changes the number the artifact reports.

    `committed_bytes()` totals every committed artifact. Including day 21's
    own file makes that a fixpoint: each write produces a different total, so
    the next write produces a different file, and `--check` can never agree
    with what is on disk. It is the sort of bug that looks like flakiness.
    """

    def test_day_21_is_absent_from_its_own_inventory(self):
        assert ledger.SELF == 21
        assert 21 not in {row.day for row in ledger.inventory()}

    def test_every_other_day_in_the_plan_is_present(self):
        assert {row.day for row in ledger.inventory()} == {
            p.day for p in projects() if p.day != ledger.SELF
        }

    def test_the_total_is_what_git_actually_stores(self):
        """The size of an artifact is not a property of the artifact.

        `.gitattributes` sets `* text=auto`, this repository is written on
        Windows and built on Linux, so every JSON artifact is one byte per
        line longer in a Windows working tree than in the index. Measuring
        with `st_size` put a number in the artifact that the machine checking
        it could never reproduce — `--check` would have failed on every push,
        reporting drift with nothing drifting.

        So the total is compared against git's own record rather than against
        a constant. If the two agree, the figure means the same thing
        everywhere.
        """
        import subprocess

        try:
            listing = subprocess.run(
                ["git", "ls-files", "-s", "--", "src/convo/projects/*/artifacts/*"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):  # pragma: no cover
            pytest.skip("git is not available here")

        blobs = [line.split()[1] for line in listing.splitlines() if line.strip()]
        if not blobs:  # pragma: no cover - a checkout with no artifacts committed
            pytest.skip("no artifacts are committed")
        sizes = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objectsize)"],
            input="\n".join(blobs),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        stored = sum(int(line) for line in sizes.split())
        assert ledger.committed_bytes() == stored

    def test_the_total_does_not_move_when_the_artifact_is_rewritten(self, tmp_path):
        before = ledger.committed_bytes()
        build.main([])
        assert ledger.committed_bytes() == before

    def test_and_the_rebuild_is_byte_stable(self):
        first = build.ARTIFACT_FILE.read_bytes()
        build.main([])
        assert build.ARTIFACT_FILE.read_bytes() == first


class TestTheArtifactIsATripwire:
    def test_check_passes_against_what_is_committed(self):
        assert build.main(["--check"]) == 0

    def test_a_moved_figure_is_reported_by_name(self, fresh):
        stale = json.loads(json.dumps(fresh))
        stale["claims"][0]["figure"] = "something else"
        complaints = build.differences(stale, fresh)
        assert any(fresh["claims"][0]["key"] in c for c in complaints), complaints

    def test_a_changed_total_is_reported_too(self, fresh):
        stale = json.loads(json.dumps(fresh))
        stale["committed_bytes"] = 1
        assert "committed_bytes: changed" in build.differences(stale, fresh)

    def test_an_unchanged_payload_reports_nothing(self, fresh):
        assert build.differences(json.loads(json.dumps(fresh)), fresh) == []

    def test_the_loader_reads_back_what_the_build_wrote(self, stored, fresh):
        assert stored.shipped == fresh["shipped"]
        assert stored.committed_bytes == fresh["committed_bytes"]
        assert len(stored.claims) == len(fresh["claims"])
        assert stored.tally == fresh["tally"]


class TestTheLoaderStaysLight:
    def test_it_imports_no_other_day(self):
        """The build imports fourteen days' results; the tab must not.

        A tab that loaded every artifact in the repository to draw one table
        would make the hub's first paint depend on all of them.
        """
        from convo.projects.day21_capstone import results

        tree = ast.parse(Path(results.__file__).read_text(encoding="utf-8"))
        imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        ]
        assert not [m for m in imported if "projects.day" in m], imported


class TestTheCostTableSaysWhatItCost:
    def test_every_row_reads_zero(self, stored):
        assert {row["dollars"] for row in stored.costs} == {"$0"}

    def test_and_every_row_says_what_that_cost_instead(self, stored):
        """The column that makes the table worth printing rather than a boast."""
        for row in stored.costs:
            assert row["lost"].strip(), row["wanted"]

    def test_the_gaps_are_recorded_with_reasons(self, stored):
        assert stored.gaps
        for gap in stored.gaps:
            assert gap["what"].strip() and gap["why"].strip()

    def test_the_unbuilt_day_is_named(self, stored):
        """Day 17 was skipped and nothing in the repository said so."""
        unshipped = [row["day"] for row in stored.inventory if not row["shipped"]]
        assert unshipped == [17]
        assert any("17" in gap["what"] for gap in stored.gaps)


class TestThePipelineMeasuresTheMiddle:
    def test_exactly_one_stage_is_measured_here(self, stored):
        measured = [s for s in stored.pipeline if s["measured"]]
        assert len(measured) == 1
        assert measured[0]["name"] == "committed artifacts"

    def test_the_two_unmeasured_stages_are_the_ones_outside_git(self, stored):
        asserted = {s["name"] for s in stored.pipeline if not s["measured"]}
        assert asserted == {"laptop", "serving image"}

    def test_the_image_gets_a_ceiling_rather_than_a_size(self, stored):
        """A weaker claim that CI checks beats a precise one that nothing does."""
        image = next(s for s in stored.pipeline if s["name"] == "serving image")
        assert str(stored.image_ceiling_mb) in image["size"]
        assert "ceiling" in image["note"]

    def test_the_blocked_modules_are_the_ones_ci_refuses(self, stored):
        workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        for module in stored.blocked_in_image:
            assert module in workflow, module
