"""Run every retriever over every chunking and store the rankings.

    uv sync --extra train
    uv run python -m convo.projects.day09_retrieval_evals.build

The artifact holds the top-10 chunk ids per (strategy, retriever, question)
plus the derived relevant set. Metrics are then recomputed from it in pure
Python, so the tab can show hit@k, MRR and nDCG for any cutoff without a model
— and so the numbers in the README can be regenerated and checked rather than
transcribed from a terminal I happened to be looking at.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.chunking import STRATEGIES, chunk_corpus, load_corpus
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day09_retrieval_evals.gold import ALL, GOLD, locate, relevant

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEPTH = 10
"""How many candidates each retriever returns, and how deep the reranker goes."""

log = get_logger("day09.build")


def load_reranker(model_id: str = CROSS_ENCODER):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32).eval()
    log.info(
        "reranker loaded",
        extra={"model": model_id, "seconds": round(time.perf_counter() - started, 1)},
    )
    return model, tokenizer


def rerank(
    model, tokenizer, query: str, candidate_ids: list[str], texts: dict[str, str]
) -> tuple[list[str], float]:
    """Reorder candidates by cross-encoder relevance. Returns (order, ms)."""
    import torch

    if not candidate_ids:
        return [], 0.0
    started = time.perf_counter()
    encoded = tokenizer(
        [query] * len(candidate_ids),
        [texts[c] for c in candidate_ids],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits.squeeze(-1)
    order = torch.argsort(logits, descending=True).tolist()
    elapsed = (time.perf_counter() - started) * 1000
    return [candidate_ids[i] for i in order], elapsed


def main() -> int:
    setup()
    documents = load_corpus()
    by_id = {d.doc_id: d for d in documents}

    encoder, encoder_tok = load_encoder(BASE_MODEL)
    question_vectors = embed([g.question for g in ALL], encoder, encoder_tok, max_len=64)
    reranker, reranker_tok = load_reranker()

    payload: dict[str, object] = {
        "encoder": BASE_MODEL,
        "reranker": CROSS_ENCODER,
        "depth": DEPTH,
        "questions": [
            {
                "question": g.question,
                "doc_id": g.doc_id,
                "quote": g.quote,
                "hard": g not in GOLD,
            }
            for g in ALL
        ],
        "strategies": {},
    }
    rerank_times: list[float] = []

    for strategy in STRATEGIES:
        chunks = chunk_corpus(documents, strategy.name)
        texts = {c.chunk_id: c.text for c in chunks}
        dense = DenseIndex(chunks, embed([c.text for c in chunks], encoder, encoder_tok, 256))
        lexical = BM25(chunks)

        per_question = []
        for index, gold in enumerate(ALL):
            span = locate(gold, by_id[gold.doc_id])
            relevant_ids = sorted(relevant(chunks, span))

            dense_ids = [h.chunk.chunk_id for h in dense.search(question_vectors[index], DEPTH)]
            bm25_ids = [h.chunk.chunk_id for h in lexical.search(gold.question, DEPTH)]
            hybrid_ids = reciprocal_rank_fusion({"dense": dense_ids, "bm25": bm25_ids})
            reranked_ids, elapsed = rerank(
                reranker, reranker_tok, gold.question, hybrid_ids[:DEPTH], texts
            )
            rerank_times.append(elapsed)

            per_question.append(
                {
                    "question": gold.question,
                    "relevant": relevant_ids,
                    "rankings": {
                        "dense": dense_ids,
                        "bm25": bm25_ids,
                        "hybrid": hybrid_ids[:DEPTH],
                        "hybrid+rerank": reranked_ids,
                    },
                }
            )

        payload["strategies"][strategy.name] = {  # type: ignore[index]
            "chunks": len(chunks),
            "describe": strategy.describe,
            "results": per_question,
        }
        log.info("scored", extra={"strategy": strategy.name, "chunks": len(chunks)})

    payload["rerank_ms_median"] = round(sorted(rerank_times)[len(rerank_times) // 2], 1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "rankings.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "artifact written",
        extra={
            "path": str(path),
            "kilobytes": round(path.stat().st_size / 1024),
            "rerank_ms_median": payload["rerank_ms_median"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
