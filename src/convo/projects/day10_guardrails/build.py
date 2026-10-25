"""Retrieve, rerank and cite for every probe, then store the evidence.

    uv sync --extra train
    uv run python -m convo.projects.day10_guardrails.build

Stores the top candidate and its reranker score per question, so the threshold
can be swept in the tab without a model. The red-team prompts go through the
same retrieval path as everything else — a guardrail that only runs on inputs
somebody labelled as attacks is not a guardrail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.answer import cite_in_document
from convo.projects.day08_rag_cites.chunking import chunk_corpus, load_corpus
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day09_retrieval_evals.build import DEPTH, load_reranker, rerank
from convo.projects.day09_retrieval_evals.gold import ALL as GOLD_ALL
from convo.projects.day10_guardrails import pii
from convo.projects.day10_guardrails.probes import RED_TEAM, UNANSWERABLE

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
STRATEGY = "structural"
"""Day 8 measured this as the one that cites presentably."""

log = get_logger("day10.build")


def main() -> int:
    setup()
    documents = load_corpus()
    by_id = {d.doc_id: d for d in documents}
    chunks = chunk_corpus(documents, STRATEGY)
    by_cid = {c.chunk_id: c for c in chunks}
    texts = {c.chunk_id: c.text for c in chunks}

    encoder, encoder_tok = load_encoder(BASE_MODEL)
    dense = DenseIndex(chunks, embed([c.text for c in chunks], encoder, encoder_tok, 256))
    lexical = BM25(chunks)
    reranker, reranker_tok = load_reranker()

    # The red-team prompts are redacted before retrieval, exactly as the
    # pipeline does it, so a card number never reaches the encoder.
    def prepared(question: str) -> str:
        redacted, _ = pii.redact(question)
        return redacted

    probes = (
        [("answerable", g.question, g.doc_id) for g in GOLD_ALL]
        + [("unanswerable", p.question, None) for p in UNANSWERABLE]
        + [("red_team", c.prompt, None) for c in RED_TEAM]
    )
    queries = [prepared(q) for _, q, _ in probes]
    query_vectors = embed(queries, encoder, encoder_tok, max_len=64)

    started = time.perf_counter()
    records = []
    for index, (kind, original, expect_doc) in enumerate(probes):
        query = queries[index]
        dense_ids = [h.chunk.chunk_id for h in dense.search(query_vectors[index], DEPTH)]
        bm25_ids = [h.chunk.chunk_id for h in lexical.search(query, DEPTH)]
        fused = reciprocal_rank_fusion({"dense": dense_ids, "bm25": bm25_ids})[:DEPTH]
        reranked, _ = rerank(reranker, reranker_tok, query, fused, texts)

        record = {
            "kind": kind,
            "question": original,
            "query": query,
            "expect_doc": expect_doc,
            "chunk_id": None,
            "doc_id": None,
            "score": None,
            "citation": None,
        }
        if reranked:
            top = by_cid[reranked[0]]
            import torch

            encoded = reranker_tok(
                [query],
                [texts[top.chunk_id]],
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                score = float(reranker(**encoded).logits.squeeze())

            citation = cite_in_document(top, query, by_id[top.doc_id])
            record.update(
                chunk_id=top.chunk_id,
                doc_id=top.doc_id,
                score=round(score, 4),
                citation=citation.as_dict() if citation else None,
            )
        records.append(record)

    payload = {
        "strategy": STRATEGY,
        "encoder": BASE_MODEL,
        "chunks": len(chunks),
        "records": records,
        "red_team": [
            {"prompt": c.prompt, "expect": c.expect, "attack": c.attack, "note": c.note}
            for c in RED_TEAM
        ],
        "unanswerable": [{"question": p.question, "why": p.why} for p in UNANSWERABLE],
        "built_seconds": round(time.perf_counter() - started, 1),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "evidence.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "path": str(path),
            "kilobytes": round(path.stat().st_size / 1024),
            "answerable": sum(r["kind"] == "answerable" for r in records),
            "unanswerable": sum(r["kind"] == "unanswerable" for r in records),
            "red_team": sum(r["kind"] == "red_team" for r in records),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
