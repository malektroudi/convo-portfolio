"""HTML for the static site. This module never reads an artifact.

Every figure arrives from `data.py` already formatted. A test asserts there is
no metric-shaped numeric literal anywhere in this file, which is the mechanical
version of the rule the whole repo runs on: a number on a page must be one the
build produced, not one somebody typed while looking at a terminal.

Hand-written prose is fine and expected. Prose is an argument; a number is a
claim about a measurement, and claims get checked.

No framework, no build step, no fonts fetched from anywhere. The page is one
file of HTML with inline CSS, which loads instantly, works with JavaScript
off, and will still render in five years — none of which is true of the
Gradio hub it renders the same artifacts as.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass

from convo.site.data import DayPage, Site, Stat, Table

TONE = {"good": "var(--good)", "bad": "var(--bad)", "plain": "var(--ink)"}

CSS = """
:root {
  --bg: #ffffff; --ink: #16181d; --muted: #5b6270; --line: #e3e6ec;
  --good: #0f7d57; --bad: #b4342f; --accent: #2a4fd6; --card: #f7f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --ink: #e9ecf1; --muted: #9aa3b2; --line: #272b33;
    --good: #46c08c; --bad: #e8736d; --accent: #7f9bff; --card: #1b1e24;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.62 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 0 16px 96px; }
a { color: var(--accent); }
header.top { border-bottom: 1px solid var(--line); margin-bottom: 36px; }
header.top .wrap { padding-top: 56px; padding-bottom: 32px; }
h1 { font-size: 2rem; line-height: 1.22; margin: 0 0 12px; letter-spacing: -0.02em; }
h2 { font-size: 1.28rem; margin: 44px 0 10px; letter-spacing: -0.01em; }
h3 { font-size: 1rem; margin: 28px 0 8px; }
p { margin: 0 0 14px; }
.lede { font-size: 1.08rem; color: var(--muted); max-width: 62ch; }
.muted { color: var(--muted); }
.small { font-size: 0.88rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
.stats { display: flex; flex-wrap: wrap; gap: 26px; margin: 22px 0 10px; }
.stat .v { font-size: 1.6rem; font-weight: 650; letter-spacing: -0.02em; }
.stat .l { font-size: 0.8rem; color: var(--muted); max-width: 22ch; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; margin: 6px 0 4px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
th { font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em;
     color: var(--muted); font-weight: 600; }
tr.hl td { font-weight: 650; }
.tnote { font-size: 0.86rem; color: var(--muted); max-width: 66ch; margin: 8px 0 26px; }
.caveat { border-left: 2px solid var(--line); padding: 2px 0 2px 14px;
          color: var(--muted); font-size: 0.9rem; max-width: 66ch; margin: 18px 0 0; }
.days { list-style: none; padding: 0; margin: 10px 0 0; }
.days li { border-bottom: 1px solid var(--line); }
.days a { display: flex; gap: 14px; padding: 13px 0; text-decoration: none; color: inherit; }
.days a:hover .t { color: var(--accent); }
.days .n { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 2.4em;
           font-size: 0.9rem; padding-top: 2px; }
.days .t { font-weight: 560; }
.days .d { color: var(--muted); font-size: 0.88rem; }
.days li.planned a { cursor: default; }
.days li.planned .t { color: var(--muted); font-weight: 450; }
/* A grid rather than a stack. Day 21 took the index from three findings to
   seven, and seven full-width cards pushed the list of days off the first
   two screens. auto-fit collapses to one column on a phone. */
.findings { display: grid; gap: 12px; margin: 0 0 12px;
            grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); }
.finding { background: var(--card); border: 1px solid var(--line); border-radius: 9px;
           padding: 15px 17px; margin: 0; }
.finding .v { font-size: 1.22rem; font-weight: 650; letter-spacing: -0.02em; }
.finding .h { font-weight: 560; margin: 3px 0 6px; }
.finding .b { font-size: 0.9rem; color: var(--muted); }
.finding .k { font-size: 0.66rem; letter-spacing: 0.06em; text-transform: uppercase;
              color: var(--muted); display: block; margin-bottom: 2px; }
.finding.testimony { border-color: #c98a16; }
.crumb { font-size: 0.88rem; margin: 0 0 20px; }
.status { background: var(--card); border: 1px solid var(--line); border-radius: 9px;
          padding: 14px 17px; font-size: 0.9rem; color: var(--muted); margin: 26px 0 0; }
.status strong { color: var(--ink); }
footer { border-top: 1px solid var(--line); margin-top: 56px; padding-top: 20px;
         font-size: 0.86rem; color: var(--muted); }
.demo { background: var(--card); border: 1px solid var(--line); border-radius: 9px;
        padding: 16px 18px; margin: 18px 0 8px; }
.demo input[type=range] { width: 100%; accent-color: var(--accent); margin: 10px 0 4px; }
.demo .row { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 12px; }
.demo .row div span { display: block; }
.demo .k { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.04em;
           color: var(--muted); }
.demo .n { font-size: 1.32rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.demo .verdict { font-size: 0.9rem; margin-top: 12px; }
@media (max-width: 620px) {
  h1 { font-size: 1.6rem; } .wrap { padding: 0 16px 64px; }
  .stats { gap: 18px; } table { font-size: 0.86rem; }
  th, td { padding: 6px 7px; }
}
"""


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


@dataclass(frozen=True)
class Finding:
    value: str
    headline: str
    body: str
    day: int
    kind: str = ""
    """What sort of number this is -- `derived`, `pinned`, `testimony`. Shown
    on the card, because day 21 established that a figure nothing can
    recompute belongs on a front page only if it says so."""
    means: str = ""


FAVICON = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
    "<text y='13' font-size='13'>&#9636;</text></svg>"
)
"""Inline so the page makes no network request at all, not even for an icon."""


def _page(title: str, description: str, body: str, depth: int = 0) -> str:
    root = "../" if depth else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="{FAVICON}">
<style>{CSS}</style>
</head>
<body>
{body}
<div class="wrap"><footer>
Built from the committed artifacts by <code>python -m convo.site.build</code> &mdash;
every figure on this site is read from the file that produced it.
<a href="{root}index.html">Index</a>
</footer></div>
</body>
</html>
"""


def _stats(stats: tuple[Stat, ...]) -> str:
    if not stats:
        return ""
    cells = "".join(
        f'<div class="stat"><div class="v" style="color:{TONE.get(s.tone, TONE["plain"])}">'
        f'{esc(s.value)}</div><div class="l">{esc(s.label)}</div></div>'
        for s in stats
    )
    return f'<div class="stats">{cells}</div>'


def _table(table: Table) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in table.headers)
    body = ""
    for index, row in enumerate(table.rows):
        cls = ' class="hl"' if table.highlight == index else ""
        cells = "".join(f"<td>{esc(c)}</td>" for c in row)
        body += f"<tr{cls}>{cells}</tr>"
    note = f'<p class="tnote">{esc(table.note)}</p>' if table.note else ""
    return (
        f"<h3>{esc(table.caption)}</h3>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}"
    )


def day_page(page: DayPage, extra: str = "") -> str:
    project = page.project
    title = f"Day {project.day}: {project.title}"
    body = [
        '<header class="top"><div class="wrap">',
        '<p class="crumb"><a href="../index.html">&larr; All days</a></p>',
        f"<h1>{esc(title)}</h1>",
        f'<p class="lede">{esc(page.summary or project.brief)}</p>',
        "</div></header>",
        '<div class="wrap">',
        _stats(page.stats),
    ]
    if page.no_artifact:
        body.append(f'<p class="muted">{esc(page.no_artifact)}</p>')
    body.extend(_table(t) for t in page.tables)
    if extra:
        body.append(extra)
    if page.caveat:
        body.append(
            f'<p class="caveat"><strong>What this does not show.</strong> {esc(page.caveat)}</p>'
        )
    body.append("</div>")
    return _page(title, page.summary or project.brief, "\n".join(body), depth=1)


def index_page(site: Site, findings: list[Finding], status: str) -> str:
    cards = "".join(
        f'<div class="finding {esc(f.kind)}">'
        f'<span class="k" title="{esc(f.means)}">Day {f.day:02d} &middot; {esc(f.kind)}</span>'
        f'<div class="v">{esc(f.value)}</div>'
        f'<div class="h">{esc(f.headline)}</div>'
        f'<div class="b">{esc(f.body)} '
        f'<a href="days/day{f.day:02d}.html">Day {f.day}</a></div></div>'
        for f in findings
    )

    rows = ""
    for page in site.pages:
        project = page.project
        if page.shipped:
            rows += (
                f'<li><a href="days/day{project.day:02d}.html">'
                f'<span class="n">{project.day:02d}</span>'
                f'<span><span class="t">{esc(project.title)}</span><br>'
                f'<span class="d">{esc(project.brief)}</span></span></a></li>'
            )
        else:
            rows += (
                f'<li class="planned"><a><span class="n">{project.day:02d}</span>'
                f'<span><span class="t">{esc(project.title)}</span><br>'
                f'<span class="d">{esc(project.brief)}</span></span></a></li>'
            )

    shipped = len(site.shipped)
    total = len(site.pages)
    body = f"""<header class="top"><div class="wrap">
<h1>Conversational AI engineering, measured</h1>
<p class="lede">{shipped} of {total} projects, each one built to produce a
number rather than a demo &mdash; and to report that number when it is
unflattering. Every figure on this site is recomputed from the artifact that
produced it, and the one that cannot be says so on its own card.</p>
</div></header>
<div class="wrap">
<h2>{len(findings)} findings worth the time</h2>
<div class="findings">{cards}</div>
<h2>Every day</h2>
<ul class="days">{rows}</ul>
<div class="status">{status}</div>
</div>"""
    return _page(
        "Conversational AI engineering, measured",
        "A measured conversational AI portfolio: retrieval that cites, "
        "guardrails that refuse, voice under a latency budget, escalation "
        "that hands over in time.",
        body,
    )


def threshold_demo(evidence: dict, default: float, low: float, high: float, step: float) -> str:
    """The one interactive piece, and the signature moment of the repo.

    Every number it needs — bounds, starting position, step — arrives as an
    argument. An earlier version opened the slider at a hand-typed -6.25,
    which was correct until day 10 was next rebuilt, and would then have been
    a stale figure on a page whose whole claim is that it has none.

    Day 10's threshold recomputed in the browser from the same scores the
    Gradio tab uses. It is here because the tradeoff is the single idea this
    portfolio keeps arriving at, and a slider argues it in a way a table
    cannot: nothing is free, and the knob has two costs that move apart.
    """
    payload = json.dumps(evidence, separators=(",", ":"))
    return f"""
<h3>Move the threshold</h3>
<p class="tnote">Recomputed in your browser from the same stored reranker
scores the hub uses. Nothing is fetched and nothing is precomputed per
position &mdash; the arithmetic is the arithmetic.</p>
<div class="demo">
  <input id="t" type="range" min="{low}" max="{high}" step="{step}" value="{default}">
  <div class="row">
    <div><span class="k">threshold</span><span class="n" id="tv"></span></div>
    <div><span class="k">answered correctly</span><span class="n" id="acc"></span></div>
    <div><span class="k">unanswerable refused</span><span class="n" id="ref"></span></div>
    <div><span class="k">answerable refused</span><span class="n" id="fr"></span></div>
  </div>
  <p class="verdict" id="verdict"></p>
</div>
<script>
const D = {payload};
const el = id => document.getElementById(id);
function render() {{
  const t = parseFloat(el('t').value);
  let right = 0, refusedAnswerable = 0;
  for (const row of D.answerable) {{
    if (row.s < t || !row.c) refusedAnswerable++;
    else if (row.ok) right++;
  }}
  let refusedUnanswerable = 0;
  for (const row of D.unanswerable) {{
    if (row.s < t || !row.c) refusedUnanswerable++;
  }}
  const a = D.answerable.length, u = D.unanswerable.length;
  const pct = x => Math.round(x * 100) + '%';
  el('tv').textContent = t.toFixed(2);
  el('acc').textContent = pct(right / a);
  el('ref').textContent = pct(refusedUnanswerable / u);
  el('fr').textContent = pct(refusedAnswerable / a);
  const leaked = u - refusedUnanswerable;
  el('verdict').textContent = leaked === 0
    ? 'Nothing unanswerable gets answered. That costs '
      + refusedAnswerable + ' of ' + a + ' answerable questions, declined anyway.'
    : leaked + ' question' + (leaked === 1 ? '' : 's')
      + ' the corpus cannot answer would be answered regardless.';
}}
el('t').addEventListener('input', render);
render();
</script>
"""
