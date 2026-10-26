"""Does the reranker's value come from the corpus being small?

Two explanations survived the reranker comparison, and only one is testable
with what is here. Neither the model's training data nor its size accounts for
reranking being worth 63% of available headroom on fourteen articles and
nothing on 5,183. What differs is the **shortlist**: ten candidates drawn from
49 chunks are mostly obviously wrong, while ten drawn from 5,183 are ten
documents that BM25 *and* an embedding model both liked — every one a hard
negative. A cross-encoder earns its cost separating close calls, and a tiny
corpus cannot manufacture close calls.

So: hold everything constant except how many documents are available to
compete, and watch the lift.

  queries   the same 42 throughout, sampled once
  labels    the same, and every judged document is present at every size
  model     the same cross-encoder day 9 used
  varying   corpus size, 49 documents up to all 5,183

The smallest size is 49 on purpose. Day 9's structural chunking produces 49
chunks for 42 questions, so the first point on this curve is SciFact rebuilt
at exactly the scale of the corpus I wrote. If the lift reappears there and
decays as documents are added, the explanation is corpus size and day 9's
headline is an artefact of a small corpus rather than a fact about reranking.

The corpus is embedded **once** at full size and sliced per condition, so
every document carries an identical vector at every corpus size — a
re-embedding per step would let a changed vector masquerade as a changed
difficulty.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion, score_runs
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.index import BM25, DenseIndex
from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
DEPTH = 10
QUERIES = 42
"""Matched to day 9's question count, so the smallest condition is a true
like-for-like rebuild of the restaurant corpus at SciFact's difficulty."""

SIZES = (49, 100, 250, 500, 1000, 2500, 5183)
SEED = 13

log = get_logger("benchmark.distractors")


def _reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER)
    model = AutoModelForSequenceClassification.from_pretrained(
        CROSS_ENCODER, dtype=torch.float32
    ).eval()
    return model, tokenizer


def run(sizes: tuple[int, ...] = SIZES) -> dict:
    import torch

    from convo.benchmark.beir import load as load_beir

    benchmark = load_beir("scifact")
    rng = random.Random(SEED)

    # One fixed query sample, drawn before anything is measured.
    chosen = sorted(benchmark.queries)
    rng.shuffle(chosen)
    chosen = sorted(chosen[:QUERIES])
    needed: set[str] = set()
    for query_id in chosen:
        needed |= benchmark.relevant[query_id]

    by_id = {c.chunk_id: c for c in benchmark.chunks}
    judged = [by_id[c] for c in sorted(needed) if c in by_id]
    fillers = [c for c in benchmark.chunks if c.chunk_id not in needed]
    rng.shuffle(fillers)

    log.info(
        "sample drawn",
        extra={"queries": len(chosen), "judged_documents": len(judged)},
    )

    encoder, tokenizer = load_encoder(BASE_MODEL)
    started = time.perf_counter()
    all_chunks = judged + fillers
    vectors = embed([c.text for c in all_chunks], encoder, tokenizer, max_len=256)
    log.info(
        "corpus embedded once",
        extra={"chunks": len(all_chunks), "seconds": round(time.perf_counter() - started)},
    )
    query_vectors = embed([benchmark.queries[q] for q in chosen], encoder, tokenizer, max_len=64)
    model, reranker_tok = _reranker()

    rows = []
    for size in sizes:
        if size < len(judged):
            log.info("size skipped", extra={"size": size, "needs": len(judged)})
            continue
        chunks = all_chunks[:size]
        subset_vectors = np.asarray(vectors[:size])
        bm25 = BM25(chunks)
        dense = DenseIndex(chunks, subset_vectors)
        texts = {c.chunk_id: c.text for c in chunks}

        fused_runs, reranked_runs = {}, {}
        for position, query_id in enumerate(chosen):
            question = benchmark.queries[query_id]
            want = benchmark.relevant[query_id]
            lexical = [h.chunk.chunk_id for h in bm25.search(question, k=DEPTH)]
            dense_ids = [h.chunk.chunk_id for h in dense.search(query_vectors[position], k=DEPTH)]
            fused = reciprocal_rank_fusion({"bm25": lexical, "dense": dense_ids})[:DEPTH]
            fused_runs[query_id] = (fused, want)

            if fused:
                encoded = reranker_tok(
                    [question] * len(fused),
                    [texts[c] for c in fused],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    logits = model(**encoded).logits.squeeze(-1)
                order = torch.argsort(logits.reshape(-1), descending=True).tolist()
                reranked_runs[query_id] = ([fused[i] for i in order], want)
            else:
                reranked_runs[query_id] = ([], want)

        fusion = score_runs("hybrid", fused_runs, k=5).as_dict()
        reranked = score_runs("hybrid+rerank", reranked_runs, k=5).as_dict()
        headroom = round(fusion["hit_at_5"] - fusion["mrr"], 4)
        delta = round(reranked["mrr"] - fusion["mrr"], 4)
        rows.append(
            {
                "documents": len(chunks),
                "fusion": fusion,
                "reranked": reranked,
                "delta_mrr": delta,
                "headroom": headroom,
                "captured": round(delta / headroom, 4) if headroom else 0.0,
            }
        )
        log.info(
            "size measured",
            extra={
                "documents": len(chunks),
                "fusion_mrr": fusion["mrr"],
                "delta": delta,
                "captured": rows[-1]["captured"],
            },
        )

    return {
        "queries": len(chosen),
        "judged_documents": len(judged),
        "depth": DEPTH,
        "reranker": CROSS_ENCODER,
        "seed": SEED,
        "sizes": [r["documents"] for r in rows],
        "rows": rows,
        "verdict": verdict(rows),
    }


COMPARABLE_HEADROOM = 0.10
"""How much room fusion must leave for a condition to resemble day 9's.

The restaurant corpus left 0.178 MRR on the table for a reranker to win. A
condition where fusion has already solved the task is not a harder version of
day 9 at a different size, it is a different experiment.
"""


def verdict(rows: list[dict], queries: int = QUERIES) -> str:
    """Did the lift decay as the corpus grew? Derived, and willing to say no.

    Three ways this experiment can fail to answer its own question, all of
    which produce a plausible-looking table:

      too small      at n queries, MRR moves by 0.5/n per rank change, so a
                     "lift" can be one query
      not comparable if fusion already solves the smallest corpus, that
                     condition is not day 9 at a different scale
      not monotonic  a decay hypothesis needs a decay, not two bumps

    Each is checked before any conclusion is offered, because an inconclusive
    experiment reported as a result is worse than no experiment.
    """
    from convo.benchmark.rerankers import material

    if len(rows) < 3:
        return "not enough sizes to say anything"

    band = material(queries)
    small, large = rows[0], rows[-1]

    problems = []
    if small["headroom"] < COMPARABLE_HEADROOM:
        problems.append(
            f"the smallest corpus leaves only {small['headroom']:.3f} MRR of "
            f"headroom against the restaurant corpus's 0.178, so it is not a "
            f"harder version of day 9 at the same scale — fusion has very "
            f"nearly solved it"
        )

    material_rows = [r for r in rows if abs(r["delta_mrr"]) >= band]
    if not material_rows:
        problems.append(
            f"no corpus size shows a change larger than {band:.3f} MRR, which "
            f"is three queries out of {queries} moving one rank"
        )
    else:
        captured = [r["captured"] for r in rows]
        descending = all(a >= b - 0.05 for a, b in zip(captured, captured[1:], strict=False))
        if not descending:
            peak = max(rows, key=lambda r: r["delta_mrr"])
            problems.append(
                f"the lift does not decay with corpus size — it peaks at "
                f"{peak['documents']} documents, between conditions that show less"
            )

    if problems:
        return (
            "inconclusive. " + "; ".join(problems) + ". The distractor "
            "hypothesis is neither supported nor refuted by this run"
        )

    if small["delta_mrr"] >= band > large["delta_mrr"]:
        return (
            "supports the distractor hypothesis: on the same queries, the same "
            "labels and the same model, reranking helps at a corpus the size of "
            "day 9's and stops helping once there are enough documents to make "
            "the shortlist competitive"
        )
    if small["delta_mrr"] >= band and large["delta_mrr"] >= band:
        return (
            "contradicts it: reranking helps at both ends, so corpus size is "
            "not what separated the two corpora"
        )
    return "no material effect at either end"


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)

    payload = run(tuple(args.sizes) if args.sizes else SIZES)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "distractors.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info("artifact written", extra={"path": str(out), "verdict": payload["verdict"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
