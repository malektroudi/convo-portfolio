"""Day 8's tab: a retrieval debugger where the citation is checkable.

The point of the page is the highlighted span. Anyone can print a chunk and
call it a source; showing the exact characters inside the original article,
with the offsets written next to them, is a claim a reader can falsify in two
seconds. That is the difference day 8 is about.
"""

from __future__ import annotations

import html

import gradio as gr
import pandas as pd

from convo.projects.day08_rag_cites.answer import cite_in_document
from convo.projects.day08_rag_cites.results import RagIndex, load

ACCENT = "#2a78d6"
GOOD = "#0ca30c"
WARN = "#b4530a"
CRITICAL = "#d03b3b"

MISSING = """
### Day 08 · RAG that actually cites

No index artifact found.

```bash
uv sync --extra train
uv run python -m convo.projects.day08_rag_cites.build
```

That chunks the help centre three ways, embeds every chunk and the question
set, and writes a ~190 KB artifact. Reload the hub afterwards.
"""


def _highlight(index: RagIndex, doc_id: str, start: int, end: int) -> str:
    """The whole article with the cited span marked, so the claim is checkable."""
    document = index.by_id[doc_id]
    before = html.escape(document.text[:start])
    span = html.escape(document.text[start:end])
    after = html.escape(document.text[end:])
    return (
        f"<div style='font-family:ui-monospace,monospace;font-size:0.78rem;"
        f"line-height:1.65;white-space:pre-wrap;max-height:340px;overflow:auto;"
        f"border:1px solid rgba(128,128,128,0.25);border-radius:6px;padding:12px 14px'>"
        f"<span style='opacity:0.45'>{before}</span>"
        f"<mark style='background:rgba(42,120,214,0.22);color:inherit;"
        f"border-bottom:2px solid {ACCENT};padding:1px 0'>{span}</mark>"
        f"<span style='opacity:0.45'>{after}</span></div>"
    )


def _answer_block(citation, hit, expect_doc: str | None) -> str:
    if citation is None:
        return (
            f"<div style='padding:12px 14px;border-radius:6px;"
            f"border:1px solid {WARN};background:rgba(180,83,10,0.10)'>"
            f"<strong>No sentence in the top chunk answers this.</strong><br>"
            f"<span style='font-size:0.85rem;opacity:0.8'>Retrieval returned "
            f"<code>{hit.chunk.doc_id}</code> but no span overlaps the question. "
            f"Refusing to cite is the correct behaviour — day 10 turns this into "
            f"an explicit refusal rather than an empty panel.</span></div>"
        )

    right = expect_doc is None or citation.doc_id == expect_doc
    colour = GOOD if right else CRITICAL
    verdict = (
        "expected article"
        if expect_doc and right
        else f"expected <code>{expect_doc}</code>"
        if expect_doc
        else "no expected article"
    )
    terms = ", ".join(citation.matched_terms) or "—"
    return (
        f"<div style='padding:14px 16px;border-radius:6px;"
        f"border:1px solid rgba(128,128,128,0.25)'>"
        f"<div style='font-size:0.68rem;letter-spacing:0.05em;text-transform:uppercase;"
        f"opacity:0.55;margin-bottom:6px'>cited span</div>"
        f"<div style='font-size:1.02rem;line-height:1.55'>{html.escape(citation.text)}</div>"
        f"<div style='font-size:0.76rem;opacity:0.65;margin-top:10px'>"
        f"<code>{citation.doc_id}</code> characters "
        f"<code>{citation.start}–{citation.end}</code>"
        f"{' · ' + html.escape(citation.heading) if citation.heading else ''}"
        f" · matched: {html.escape(terms)}"
        f" · <span style='color:{colour}'>{verdict}</span></div></div>"
    )


def _hits_frame(index: RagIndex, hits, question: str) -> pd.DataFrame:
    rows = []
    for hit in hits:
        citation = cite_in_document(hit.chunk, question, index.by_id[hit.chunk.doc_id])
        rows.append(
            {
                "rank": hit.rank,
                "article": hit.chunk.doc_id,
                "heading": hit.chunk.heading or "—",
                "score": round(hit.score, 3),
                "chars": f"{hit.chunk.start}–{hit.chunk.end}",
                "cited span": (citation.text[:70] + "…")
                if citation and len(citation.text) > 70
                else (citation.text if citation else "— nothing matched —"),
            }
        )
    return pd.DataFrame(rows)


def _strategy_table(index: RagIndex) -> str:
    rows = ""
    for name, strategy in index.strategies.items():
        stats = strategy.stats
        mid = stats["starts_mid_sentence"]
        share = mid / max(stats["chunks"], 1)
        colour = GOOD if not mid else CRITICAL if share > 0.25 else WARN
        rows += (
            f"<tr><td style='padding:5px 12px 5px 0'><code>{name}</code></td>"
            f"<td style='padding:5px 12px;text-align:right'>{stats['chunks']}</td>"
            f"<td style='padding:5px 12px;text-align:right'>{stats['mean_chars']:.0f}</td>"
            f"<td style='padding:5px 12px;text-align:right;color:{colour}'>"
            f"{mid} ({share:.0%})</td>"
            f"<td style='padding:5px 0;font-size:0.82rem;opacity:0.7'>"
            f"{stats['describe']}</td></tr>"
        )
    return f"""
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9rem">
<thead><tr style="border-bottom:1px solid rgba(128,128,128,0.35)">
  <th style="text-align:left;padding:4px 12px 4px 0">strategy</th>
  <th style="text-align:right;padding:4px 12px">chunks</th>
  <th style="text-align:right;padding:4px 12px">mean chars</th>
  <th style="text-align:right;padding:4px 12px">start mid-sentence</th>
  <th style="text-align:left;padding:4px 0"></th>
</tr></thead><tbody>{rows}</tbody></table></div>
"""


def _run(question: str, typed: str, strategy_name: str):
    index = load()
    if index is None:
        return "", "", pd.DataFrame()

    query = (typed or "").strip() or question
    strategy = index.strategies[strategy_name]
    vector = index.vector_for(query)

    if vector is not None:
        hits = strategy.dense.search(vector, k=4)
        note = "dense retrieval — this question's embedding ships in the artifact"
    else:
        hits = strategy.bm25.search(query, k=4)
        note = "BM25 — typed questions are matched lexically, because the hub carries no encoder"

    if not hits:
        return (
            f"<div style='padding:12px 14px;border:1px solid {WARN};border-radius:6px'>"
            f"<strong>Nothing retrieved.</strong> No chunk shares a term with that "
            f"question.</div>",
            "",
            pd.DataFrame(),
        )

    top = hits[0]
    document = index.by_id[top.chunk.doc_id]
    citation = cite_in_document(top.chunk, query, document)
    expect = next((q.expect_doc for q in index.questions if q.text == query), None)

    answer = (
        f"<p style='font-size:0.78rem;opacity:0.6;margin-bottom:8px'>{note}</p>"
        + _answer_block(citation, top, expect)
    )
    highlighted = (
        _highlight(index, citation.doc_id, citation.start, citation.end)
        if citation
        else _highlight(index, top.chunk.doc_id, top.chunk.start, top.chunk.end)
    )
    return answer, highlighted, _hits_frame(index, hits, query)


def build() -> None:
    index = load()
    if index is None:
        gr.Markdown(MISSING)
        return

    questions = [q.text for q in index.questions]
    default = questions[0]
    strategies = list(index.strategies)

    gr.Markdown(
        f"### Day 08 · RAG that actually cites\n"
        f"A {len(index.documents)}-article help centre, chunked three ways, with "
        f"citations that point at **exact character offsets** in the source. Every "
        f"chunk and every citation satisfies one invariant, asserted in the tests:\n\n"
        f"```\ndocument.text[start:end] == cited_text\n```\n\n"
        f"Most RAG demos return a chunk id and call it a source, which is "
        f"unfalsifiable — the chunk is the only record of itself. Keeping offsets "
        f"means you can highlight the span in the original and a reader can check "
        f"it. The corpus is synthetic so that it could be built with the failure "
        f"modes that matter: a near-duplicate policy pair, a superseded deposit "
        f"figure, hours split across two articles, and two questions nothing "
        f"answers."
    )

    with gr.Row():
        question = gr.Dropdown(
            choices=questions, value=default, label="Question (dense retrieval)", scale=3
        )
        strategy = gr.Radio(choices=strategies, value="structural", label="Chunking", scale=2)
    typed = gr.Textbox(
        label="…or type your own (falls back to BM25, no model needed)",
        placeholder="do you have EV charging?",
    )

    answer = gr.HTML()
    gr.Markdown("#### The span, highlighted in the article it came from")
    highlighted = gr.HTML()
    gr.Markdown("#### What retrieval returned")
    hits = gr.Dataframe(interactive=False, wrap=True)

    gr.Markdown(
        "#### Chunking changes the citation, not the retrieval\n"
        "All three strategies find the right article about equally often — 81–84% "
        "at rank 1 and 100% within the top 3. What changes is whether the span you "
        "can show a human reads as a sentence. Fixed chunking starts 48% of its "
        "chunks mid-sentence and drags markdown headings into 9% of its citations; "
        "structural chunking starts none and cites spans a sixth shorter."
    )
    gr.HTML(_strategy_table(index))

    inputs = [question, typed, strategy]
    outputs = [answer, highlighted, hits]
    for control in (question, strategy):
        control.change(_run, inputs=inputs, outputs=outputs)
    typed.submit(_run, inputs=inputs, outputs=outputs)

    first = _run(default, "", "structural")
    answer.value, highlighted.value, hits.value = first
