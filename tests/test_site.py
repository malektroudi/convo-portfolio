"""The static site: that it builds, that it is honest, and that it agrees.

Three things are worth testing about a generated portfolio site, and none of
them is "does the HTML look right".

The first is that **no figure is typed by hand**. The renderer must contain no
metric-shaped literal, because the failure mode here is not a crash — it is a
number that was true in October and is quietly wrong in November.

The second is that **the site agrees with the hub**. Both render the same
artifacts through code that shares nothing, so a disagreement means one of
them is reading the file wrong. That is the whole claim of the
artifacts-as-interface design, and it is cheap to check.

The third is that **the page works without Python**. The one interactive piece
reimplements day 10's threshold arithmetic in JavaScript, and a reimplementation
that drifts from the original is worse than no slider at all.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from convo.hub import app as hub_app
from convo.site import build as build_module
from convo.site import render
from convo.site.build import findings, generate
from convo.site.data import collect


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    summary = generate(out)
    return out, summary


class TestNoFigureIsTypedByHand:
    """The renderer must not contain a number of its own.

    Checked over the parsed syntax tree rather than the text, because the
    prose in this module legitimately quotes figures and explains what
    artifacts are — and a test that greps its own documentation is a test that
    fails for the wrong reason.
    """

    @staticmethod
    def _literals(module: str) -> list:
        tree = ast.parse(Path(module).read_text(encoding="utf-8"))
        # Docstrings are prose. Strip them, then look at what is left.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                first = node.body[0] if node.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docstrings.add(id(first.value))
        return [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and id(node) not in docstrings
        ]

    def test_the_renderer_has_no_float_literal(self):
        """A float in a renderer is a measurement that will go stale.

        Integers are allowed: `depth=1` and a slice index are structure, not
        claims. Floats are not — every one of them in a page like this is a
        rate, a score or a duration.
        """
        floats = [v for v in self._literals(render.__file__) if isinstance(v, float)]
        assert not floats, f"render.py carries its own numbers: {floats}"

    def test_the_renderer_has_no_percentage_or_millisecond_string(self):
        strings = [v for v in self._literals(render.__file__) if isinstance(v, str)]
        text = " ".join(s for s in strings if s != render.CSS)
        found = re.findall("\\d+\\s?%|\\d+\\.\\d+|\\d+\\s?ms\\b", text)
        assert not found, f"render.py quotes figures directly: {found}"

    @pytest.mark.parametrize("module", [build_module, hub_app])
    def test_neither_landing_page_quotes_a_figure(self, module):
        """The exemption that cost three stale numbers.

        This rule was only ever enforced on the renderer. The two modules that
        decide what the index and the hub's overview say were free to type
        their own, and both did: the hub's lead card had been wrong since day
        3's artifact was last rebuilt, and both described a training run whose
        artifact no longer exists. Day 21 made the findings data; this stops
        them becoming literals again.

        Megabytes are included because the hub's architecture diagram carried
        a stale one of those too.
        """
        strings = [v for v in self._literals(module.__file__) if isinstance(v, str)]
        found = re.findall("\\d+\\s?%|\\d+\\s?ms\\b|\\d+\\s?MB\\b", " ".join(strings))
        assert not found, f"{module.__name__} quotes figures directly: {found}"

    def test_the_renderer_imports_nothing_from_the_projects(self):
        """Its inputs arrive as arguments; it cannot reach an artifact."""
        tree = ast.parse(Path(render.__file__).read_text(encoding="utf-8"))
        imported = [
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not [m for m in imported if "projects" in m], imported

    def test_the_findings_point_at_days_that_exist(self):
        days = {page.day for page in collect().shipped}
        for finding in findings():
            assert finding.day in days, finding.headline


class TestItBuilds:
    def test_every_shipped_day_gets_a_page(self, built):
        out, _ = built
        for page in collect().shipped:
            assert (out / "days" / f"{page.slug}.html").exists(), page.slug

    def test_planned_days_do_not_get_a_page(self, built):
        out, _ = built
        for page in collect().planned:
            assert not (out / "days" / f"{page.slug}.html").exists(), page.slug

    def test_there_is_an_index_and_a_nojekyll(self, built):
        out, _ = built
        assert (out / "index.html").exists()
        # Without this, Pages hides anything starting with an underscore.
        assert (out / ".nojekyll").exists()

    def test_the_whole_site_is_small(self, built):
        """The serving image is three orders of magnitude bigger. That gap
        is the point of the day, and day 21 puts a checked ceiling on the
        other end of it."""
        _, summary = built
        assert summary["kilobytes"] < 400, summary

    def test_every_page_is_complete_html(self, built):
        out, _ = built
        for path in out.rglob("*.html"):
            text = path.read_text(encoding="utf-8")
            assert text.startswith("<!doctype html>"), path.name
            assert text.rstrip().endswith("</html>"), path.name
            assert text.count("<html") == text.count("</html>") == 1, path.name

    def test_no_page_fetches_anything_external(self, built):
        """No fonts, no CDN, no analytics. It must work offline and forever."""
        out, _ = built
        for path in out.rglob("*.html"):
            text = path.read_text(encoding="utf-8")
            external = re.findall(r'(?:src|href)="(https?:)?//[^"]+', text)
            assert not external, f"{path.name} reaches out to {external}"

    def test_internal_links_resolve(self, built):
        out, _ = built
        for path in out.rglob("*.html"):
            text = path.read_text(encoding="utf-8")
            for href in re.findall(r'href="([^":#]+\.html)"', text):
                assert (path.parent / href).resolve().exists(), f"{path.name} -> {href}"

    def test_rebuilding_is_clean(self, built):
        """A stale page from a deleted day must not survive a rebuild."""
        out, _ = built
        ghost = out / "days" / "day99.html"
        ghost.write_text("stale", encoding="utf-8")
        generate(out)
        assert not ghost.exists()


class TestItIsHonest:
    def test_every_day_page_says_what_it_does_not_show(self, built):
        out, _ = built
        for page in collect().shipped:
            if not page.tables:
                continue
            text = (out / "days" / f"{page.slug}.html").read_text(encoding="utf-8")
            assert "What this does not show" in text, page.slug

    def test_a_day_with_no_artifact_claims_no_figures(self):
        """Better an empty card than an invented one."""
        for page in collect().shipped:
            if page.no_artifact:
                assert not page.stats, page.slug
                assert not page.tables, page.slug

    def test_the_index_states_how_many_days_are_done(self, built):
        out, _ = built
        text = (out / "index.html").read_text(encoding="utf-8")
        site = collect()
        assert f"{len(site.shipped)} of {len(site.pages)} projects" in text


class TestItAgreesWithTheHub:
    """The site and the Gradio tabs share no display code, so agreement is
    evidence that both read the artifacts correctly rather than that one
    copied the other."""

    def test_day_ten_figures_match_the_results_module(self):
        from convo.projects.day10_guardrails.results import load

        found = load()
        if found is None:
            pytest.skip("day 10 artifact not built")
        page = next(p for p in collect().shipped if p.day == 10)
        best = found.best()
        values = {stat.value for stat in page.stats}
        assert f"{best.refusal_recall:.0%}" in values
        assert f"{best.false_refusal:.0%}" in values

    def test_day_thirteen_figures_match_the_results_module(self):
        from convo.projects.day13_escalation.results import load

        found = load()
        if found is None:
            pytest.skip("day 13 artifact not built")
        page = next(p for p in collect().shipped if p.day == 13)
        assert str(found.best_policy()["turns_late"]) in {s.value for s in page.stats}

    def test_the_day_count_matches_the_registry(self):
        import convo.projects  # noqa: F401
        from convo.core.registry import projects, shipped

        site = collect()
        assert len(site.pages) == len(projects())
        assert len(site.shipped) == len(shipped())


class TestTheBrowserSliderAgreesWithPython:
    """The slider reimplements day 10's gates in JavaScript.

    What is tested is the *payload*: whether the three fields the page ships
    per question are enough to reproduce day 10's own scorer. If they are, the
    JS — which does exactly this arithmetic — is reproducing it too. If the
    reduction ever drops a field the decision depends on, this fails.
    """

    @pytest.fixture(scope="class")
    def payload(self):
        from convo.site.build import _threshold_payload

        found = _threshold_payload()
        if found is None:
            pytest.skip("day 10 artifact not built")
        return found

    @staticmethod
    def _recompute(payload, threshold):
        answerable = payload["answerable"]
        unanswerable = payload["unanswerable"]
        right = sum(1 for r in answerable if r["s"] >= threshold and r["c"] and r["ok"])
        refused_answerable = sum(1 for r in answerable if r["s"] < threshold or not r["c"])
        refused_unanswerable = sum(1 for r in unanswerable if r["s"] < threshold or not r["c"])
        return (
            round(right / len(answerable), 4),
            round(refused_unanswerable / len(unanswerable), 4),
            round(refused_answerable / len(answerable), 4),
        )

    @pytest.mark.parametrize("threshold", [-12.0, -10.0, -7.0, -6.3085, -2.0, 0.0, 4.0])
    def test_it_reproduces_the_scorer_exactly(self, payload, threshold):
        from convo.projects.day10_guardrails.results import load

        found = load()
        metrics = found.at(threshold)
        assert self._recompute(payload, threshold) == (
            metrics.answer_accuracy,
            metrics.refusal_recall,
            metrics.false_refusal,
        )

    def test_the_slider_opens_exactly_on_the_measured_threshold(self):
        """A browser snaps `value` to the nearest `min + n*step`.

        Without aligning the floor, the slider opened 0.15 away from day 10's
        balanced-best threshold while the page said it opened on it. The
        metrics shown were identical either way, which is precisely why the
        drift would never have been noticed.
        """
        from convo.projects.day10_guardrails.results import load
        from convo.site.build import _threshold_range

        found = load()
        if found is None:
            pytest.skip("day 10 artifact not built")
        bounds = _threshold_range()
        assert bounds["default"] == round(found.best().threshold, 2)
        stops = (bounds["default"] - bounds["low"]) / bounds["step"]
        assert abs(stops - round(stops)) < 1e-9, f"default sits at stop {stops}"
        assert bounds["low"] <= bounds["default"] <= bounds["high"]

    def test_the_payload_carries_no_more_than_it_needs(self, payload):
        """Three fields per question. Shipping the transcript would be easier
        and would put the corpus on a public page by accident."""
        for group in ("answerable", "unanswerable"):
            for row in payload[group]:
                assert set(row) == {"s", "c", "ok"}, row

    def test_the_page_embeds_the_payload_rather_than_fetching_it(self, built):
        out, _ = built
        text = (out / "days" / "day10.html").read_text(encoding="utf-8")
        assert "const D = {" in text
        assert "fetch(" not in text
