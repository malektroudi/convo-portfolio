"""Run days 8 and 9's retrievers against a public benchmark, unchanged.

    uv sync --extra train
    uv run python -m convo.benchmark.run

Four retrievers, the same four day 9 compares, scored with the same
`core.retrieval` functions against labels nobody here wrote. The imports are
the point: `BM25` and `DenseIndex` come from day 8, `reciprocal_rank_fusion`
and `score_runs` from core, `rerank` from day 9. If any of them had to be
adapted to run on someone else's corpus, that would itself be the finding.

Expect the numbers to be worse than day 9's. Fourteen articles is not a
retrieval problem; 5,183 are. A benchmark that flattered the stack would be
the wrong benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from convo.benchmark.beir import DEFAULT, Benchmark, load
from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion, score_runs
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER, load_reranker, rerank

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
RETRIEVERS = ("bm25", "dense", "hybrid", "hybrid+rerank")

log = get_logger("benchmark.run")


DEPTHS = (10, 100)
"""Shortlist sizes to sweep.

Day 9 reranks the top 10, which is sensible over 49 chunks and very shallow
over 5,183 documents: a cross-encoder can only reorder what fusion already
surfaced, so a depth that suits a toy corpus may be the thing limiting it on a
real one. Reporting a negative result for the reranker without testing that
would be blaming the model for a parameter."""


def evaluate(benchmark: Benchmark, depths: tuple[int, ...] = DEPTHS) -> dict:
    """Score every retriever over every judged query, at each shortlist depth."""
    encoder, tokenizer = load_encoder(BASE_MODEL)

    started = time.perf_counter()
    vectors = embed([c.text for c in benchmark.chunks], encoder, tokenizer, max_len=256)
    log.info(
        "corpus embedded",
        extra={"chunks": len(benchmark.chunks), "seconds": round(time.perf_counter() - started)},
    )

    bm25 = BM25(benchmark.chunks)
    dense = DenseIndex(benchmark.chunks, vectors)
    reranker, reranker_tok = load_reranker()
    texts = {c.chunk_id: c.text for c in benchmark.chunks}

    query_ids = list(benchmark.queries)
    query_vectors = embed([benchmark.queries[q] for q in query_ids], encoder, tokenizer, max_len=64)

    by_depth: dict[str, dict] = {}
    for depth in depths:
        runs: dict[str, dict[str, tuple[list[str], set[str]]]] = {r: {} for r in RETRIEVERS}
        rerank_ms: list[float] = []

        for position, query_id in enumerate(query_ids):
            question = benchmark.queries[query_id]
            want = benchmark.relevant[query_id]

            lexical = [h.chunk.chunk_id for h in bm25.search(question, k=depth)]
            dense_ids = [h.chunk.chunk_id for h in dense.search(query_vectors[position], k=depth)]
            fused = reciprocal_rank_fusion({"bm25": lexical, "dense": dense_ids})[:depth]
            reranked, elapsed = rerank(reranker, reranker_tok, question, fused, texts)
            rerank_ms.append(elapsed)

            runs["bm25"][query_id] = (lexical, want)
            runs["dense"][query_id] = (dense_ids, want)
            runs["hybrid"][query_id] = (fused, want)
            runs["hybrid+rerank"][query_id] = (reranked, want)

            if (position + 1) % 100 == 0:
                log.info(
                    "scored",
                    extra={"depth": depth, "queries": position + 1, "of": len(query_ids)},
                )

        by_depth[str(depth)] = {
            "scores": {r: score_runs(r, runs[r], k=5).as_dict() for r in RETRIEVERS},
            "rerank_ms_median": round(sorted(rerank_ms)[len(rerank_ms) // 2], 1),
        }
        log.info("depth done", extra={"depth": depth})

    return {
        "benchmark": benchmark.describe(),
        "encoder": BASE_MODEL,
        "reranker": CROSS_ENCODER,
        "depths": list(depths),
        "by_depth": by_depth,
        "scores": by_depth[str(depths[0])]["scores"],
        "rerank_ms_median": by_depth[str(depths[0])]["rerank_ms_median"],
    }


def _restaurant_scores() -> dict:
    """Day 9's own numbers, for the side-by-side.

    Read from day 9's artifact rather than recomputed, so the comparison uses
    the figures that day actually published.
    """
    from convo.projects.day09_retrieval_evals.results import load as load_day9

    found = load_day9()
    if found is None:
        return {}
    strategy = found.strategies["structural"]
    return {r: strategy.score(r).as_dict() for r in RETRIEVERS}


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=DEFAULT)
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="truncate the corpus, keeping every judged document (for a smoke run)",
    )
    args = parser.parse_args(argv)

    benchmark = load(args.benchmark, limit_docs=args.limit_docs)
    payload = evaluate(benchmark)
    payload["restaurant"] = _restaurant_scores()
    payload["truncated"] = args.limit_docs is not None

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / f"{args.benchmark}.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info(
        "artifact written",
        extra={"path": str(out), "kilobytes": max(out.stat().st_size // 1024, 1)},
    )

    for depth, block in payload["by_depth"].items():
        for retriever in RETRIEVERS:
            row = block["scores"][retriever]
            log.info(
                "result",
                extra={
                    "depth": depth,
                    "retriever": retriever,
                    "hit_at_1": row["hit_at_1"],
                    "mrr": row["mrr"],
                    "ndcg_at_5": row["ndcg_at_5"],
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
