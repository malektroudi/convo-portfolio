"""Retrieve every turn under every condition and store the rankings.

    uv sync --extra train
    uv run python -m convo.projects.day11_multiturn_rag.build

Four conditions, which together answer a question one condition cannot:

  raw       the follow-up as typed — what a single-turn RAG system receives
  concat    the previous turn glued on — the cheap fix everyone tries first
  rewrite   the rules in rewrite.py
  oracle    the hand-written standalone question — the ceiling

`concat` is here because it is the honest baseline. Any rewriter beats `raw`;
the question is whether it beats gluing two strings together, which costs
nothing and needs no lexicon.

Each condition runs over three retrievers, because "does rewriting help" has a
different answer depending on what is downstream. BM25 matches words and
should care most; a cross-encoder reads the question and passage together and
may not need the help. Measuring one retriever would have let either
conclusion be asserted.

Memory is fed by whichever retriever is being evaluated, so each condition
sees the conversation its own system would have produced — including its own
mistakes propagating into later turns.
"""

from __future__ import annotations

import json
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.chunking import chunk_corpus, load_corpus
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER, DEPTH, load_reranker, rerank
from convo.projects.day09_retrieval_evals.gold import Gold, locate, relevant
from convo.projects.day11_multiturn_rag.conversations import CONVERSATIONS
from convo.projects.day11_multiturn_rag.memory import Memory
from convo.projects.day11_multiturn_rag.rewrite import (
    ALL_RULES,
    CARRYOVER,
    ELLIPSIS,
    GUARDED_RULES,
    PRONOUN,
    SUBSTITUTE,
    rewrite,
)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
STRATEGY = "structural"
"""Day 8 measured this one as the chunking that cites cleanly; day 10 uses it."""

RETRIEVERS = ("bm25", "dense", "hybrid+rerank")

# Cumulative ablations: each adds one rule to the one before it, so the table
# reads as "what did this rule buy" rather than "what did this rule do alone".
ABLATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pronoun", (PRONOUN,)),
    ("+substitute", (PRONOUN, SUBSTITUTE)),
    ("+ellipsis", (PRONOUN, SUBSTITUTE, ELLIPSIS)),
    ("+carryover", (PRONOUN, SUBSTITUTE, ELLIPSIS, CARRYOVER)),
    ("+guard (tuned)", GUARDED_RULES),
)

DEPTHS = (1, 2, 3, 5, 99)
"""Memory depths to sweep. 99 is "remember everything" on these conversations."""

log = get_logger("day11.build")


def _condition_query(
    condition: str,
    turn,
    memory: Memory,
    previous_text: str | None,
    rules: tuple[str, ...],
) -> tuple[str, str, str]:
    """Return (query, strategy, detail) for one turn under one condition."""
    if condition == "raw":
        return turn.text, "raw", ""
    if condition == "concat":
        if previous_text is None:
            return turn.text, "raw", ""
        return f"{previous_text} {turn.text}", "concat", f"glued onto {previous_text!r}"
    if condition == "oracle":
        return turn.resolved, "oracle" if turn.oracle else "raw", ""
    result = rewrite(turn.text, memory, rules)
    return result.text, result.strategy, result.detail


def run_condition(
    condition: str,
    retriever: str,
    bm25: BM25,
    dense: DenseIndex,
    encoder,
    encoder_tok,
    reranker,
    reranker_tok,
    texts: dict[str, str],
    depth: int,
    rules: tuple[str, ...],
) -> list[dict]:
    """Walk every conversation, retrieving each turn with its memory alive."""
    rows: list[dict] = []

    for conversation in CONVERSATIONS:
        memory = Memory(depth=depth)
        previous_text: str | None = None

        for index, turn in enumerate(conversation.turns):
            query, strategy, detail = _condition_query(
                condition, turn, memory, previous_text, rules
            )

            if retriever == "bm25":
                ranked = [h.chunk.chunk_id for h in bm25.search(query, k=DEPTH)]
            else:
                vector = embed([query], encoder, encoder_tok, max_len=64)[0]
                dense_ids = [h.chunk.chunk_id for h in dense.search(vector, k=DEPTH)]
                if retriever == "dense":
                    ranked = dense_ids
                else:
                    lexical = [h.chunk.chunk_id for h in bm25.search(query, k=DEPTH)]
                    fused = reciprocal_rank_fusion({"bm25": lexical, "dense": dense_ids})[:DEPTH]
                    ranked, _ = rerank(reranker, reranker_tok, query, fused, texts)

            top_doc = ranked[0].split("#")[0] if ranked else None
            rows.append(
                {
                    "conversation": conversation.name,
                    "index": index,
                    "text": turn.text,
                    "query": query,
                    "strategy": strategy,
                    "detail": detail,
                    "ranked": ranked,
                    "doc_id": turn.doc_id,
                    "standalone": turn.standalone,
                }
            )
            memory.observe(turn.text, query, top_doc)
            previous_text = turn.text

    return rows


def main() -> int:
    setup()
    documents = load_corpus()
    by_id = {d.doc_id: d for d in documents}
    chunks = chunk_corpus(documents, STRATEGY)
    texts = {c.chunk_id: c.text for c in chunks}

    encoder, encoder_tok = load_encoder(BASE_MODEL)
    chunk_vectors = embed([c.text for c in chunks], encoder, encoder_tok, max_len=256)
    bm25 = BM25(chunks)
    dense = DenseIndex(chunks, chunk_vectors)
    reranker, reranker_tok = load_reranker()

    # One relevant set per turn, derived from the gold span exactly as day 9
    # does, so a chunk counts as correct for the same reason in both days.
    relevance: dict[str, list[str]] = {}
    for conversation in CONVERSATIONS:
        for index, turn in enumerate(conversation.turns):
            span = locate(Gold(turn.text, turn.doc_id, turn.quote), by_id[turn.doc_id])
            key = f"{conversation.name}:{index}"
            found = relevant(chunks, span)
            if not found:
                raise ValueError(f"{key}: no chunk covers the gold span for {turn.text!r}")
            relevance[key] = sorted(found)

    payload: dict = {
        "strategy": STRATEGY,
        "chunks": len(chunks),
        "encoder": BASE_MODEL,
        "reranker": CROSS_ENCODER,
        "depth": DEPTH,
        "relevance": relevance,
        "conversations": [
            {
                "name": c.name,
                "note": c.note,
                "turns": [
                    {"text": t.text, "oracle": t.oracle, "doc_id": t.doc_id, "quote": t.quote}
                    for t in c.turns
                ],
            }
            for c in CONVERSATIONS
        ],
        "runs": {},
        "ablations": {},
        "depths": {},
    }

    common = (bm25, dense, encoder, encoder_tok, reranker, reranker_tok, texts)

    for condition in ("raw", "concat", "rewrite", "oracle"):
        for retriever in RETRIEVERS:
            rows = run_condition(
                condition, retriever, *common, depth=Memory().depth, rules=ALL_RULES
            )
            payload["runs"][f"{condition}|{retriever}"] = rows
        log.info("condition scored", extra={"condition": condition})

    for name, rules in ABLATIONS:
        rows = run_condition("rewrite", "bm25", *common, depth=Memory().depth, rules=rules)
        payload["ablations"][name] = rows
    log.info("ablations scored", extra={"count": len(ABLATIONS)})

    for depth in DEPTHS:
        rows = run_condition("rewrite", "bm25", *common, depth=depth, rules=ALL_RULES)
        payload["depths"][str(depth)] = rows
    log.info("depth sweep scored", extra={"depths": list(DEPTHS)})

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "turns.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={"path": str(out), "kilobytes": out.stat().st_size // 1024},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
