"""Is the reranker bad at science, or just bad? A controlled comparison.

Running day 9's stack on SciFact found that `ms-marco-MiniLM-L-6-v2` *costs*
0.013 MRR there while adding 0.086 on the restaurant corpus. The explanation
offered was domain mismatch — it is trained on web search passages, and
scientific claims against paper abstracts are not that. A plausible story is
not a result, so this tests it.

Three rerankers, chosen so the two candidate explanations come apart:

  ms-marco-MiniLM-L-6-v2    23M   MS MARCO        the original
  ms-marco-MiniLM-L-12-v2   33M   MS MARCO        same data, more capacity
  bge-reranker-base        278M   broad, incl.    different training entirely
                                  scientific text

If the L-12 model fixes SciFact, the problem was capacity and the domain story
is wrong. If only `bge` fixes it, the training data is what matters. If none
of them fixes it, the fault is in the shortlist or in me rather than in any
reranker.

**The shortlist is held fixed.** Fusion runs once per corpus and all three
models reorder the identical candidates, so the only thing varying is the
model. Re-running retrieval per model would have left an obvious confound in
a comparison whose entire purpose is to remove one.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from convo.core import get_logger, setup
from convo.core.retrieval import reciprocal_rank_fusion, score_runs
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.index import BM25, DenseIndex

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
DEPTH = 10
"""Shortlist size. The depth sweep already showed 100 is no better and ten
times the cost, so this comparison holds it at day 9's own setting."""


@dataclass(frozen=True)
class Reranker:
    model_id: str
    label: str
    trained_on: str
    params_m: int


RERANKERS: tuple[Reranker, ...] = (
    Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", "MiniLM-L6", "MS MARCO", 23),
    Reranker("cross-encoder/ms-marco-MiniLM-L-12-v2", "MiniLM-L12", "MS MARCO", 33),
    Reranker("BAAI/bge-reranker-base", "bge-base", "broad, incl. scientific", 278),
)

log = get_logger("benchmark.rerankers")


@dataclass
class Shortlists:
    """One corpus, reduced to what a reranker actually needs."""

    name: str
    order: dict[str, list[str]]
    """query_id -> the fused candidate ids, in fusion's order."""
    relevant: dict[str, set[str]]
    queries: dict[str, str]
    texts: dict[str, str]

    @property
    def baseline(self) -> dict:
        """Fusion's own score, which is what each reranker has to beat."""
        runs = {q: (self.order[q], self.relevant[q]) for q in self.order}
        return score_runs("hybrid", runs, k=5).as_dict()


def _load_model(model_id: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=torch.float32).eval()
    log.info(
        "reranker loaded",
        extra={"model": model_id, "seconds": round(time.perf_counter() - started, 1)},
    )
    return model, tokenizer


def rerank_all(shortlists: Shortlists, reranker: Reranker) -> dict:
    """Reorder every shortlist with one model and score the result."""
    import torch

    model, tokenizer = _load_model(reranker.model_id)
    runs: dict[str, tuple[list[str], set[str]]] = {}
    elapsed: list[float] = []

    for query_id, candidates in shortlists.order.items():
        if not candidates:
            runs[query_id] = ([], shortlists.relevant[query_id])
            continue
        question = shortlists.queries[query_id]
        started = time.perf_counter()
        encoded = tokenizer(
            [question] * len(candidates),
            [shortlists.texts[c] for c in candidates],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        order = torch.argsort(logits.reshape(-1), descending=True).tolist()
        elapsed.append((time.perf_counter() - started) * 1000)
        runs[query_id] = ([candidates[i] for i in order], shortlists.relevant[query_id])

    scores = score_runs(reranker.label, runs, k=5).as_dict()
    return {
        "model": reranker.model_id,
        "label": reranker.label,
        "trained_on": reranker.trained_on,
        "params_m": reranker.params_m,
        "scores": scores,
        "ms_median": round(sorted(elapsed)[len(elapsed) // 2], 1) if elapsed else 0.0,
    }


# ------------------------------------------------------------------ corpora


def restaurant_shortlists() -> Shortlists:
    """Day 9's own fused rankings, read back rather than recomputed.

    The chunk texts are regenerated from the corpus, which is deterministic
    and needs no model — day 9's artifact stores ids, and a reranker needs the
    prose behind them.
    """
    from convo.projects.day08_rag_cites.chunking import chunk_corpus, load_corpus
    from convo.projects.day09_retrieval_evals.results import load as load_day9

    found = load_day9()
    if found is None:
        raise SystemExit("day 9 has no rankings artifact; run its build first")

    strategy = found.strategies["structural"]
    chunks = chunk_corpus(load_corpus(), "structural")
    texts = {c.chunk_id: c.text for c in chunks}

    order, relevant, queries = {}, {}, {}
    for question in strategy.questions:
        key = question.question
        order[key] = list(question.rankings["hybrid"])[:DEPTH]
        relevant[key] = set(question.relevant)
        queries[key] = key
    return Shortlists("restaurant", order, relevant, queries, texts)


def scifact_shortlists(limit_docs: int | None = None) -> Shortlists:
    """Fuse BM25 and dense over SciFact once, for all three models to share."""
    from convo.benchmark.beir import load as load_beir

    benchmark = load_beir("scifact", limit_docs=limit_docs)
    encoder, tokenizer = load_encoder(BASE_MODEL)

    started = time.perf_counter()
    vectors = embed([c.text for c in benchmark.chunks], encoder, tokenizer, max_len=256)
    log.info(
        "corpus embedded",
        extra={"chunks": len(benchmark.chunks), "seconds": round(time.perf_counter() - started)},
    )

    bm25 = BM25(benchmark.chunks)
    dense = DenseIndex(benchmark.chunks, vectors)
    texts = {c.chunk_id: c.text for c in benchmark.chunks}

    query_ids = list(benchmark.queries)
    query_vectors = embed([benchmark.queries[q] for q in query_ids], encoder, tokenizer, max_len=64)

    order = {}
    for position, query_id in enumerate(query_ids):
        question = benchmark.queries[query_id]
        lexical = [h.chunk.chunk_id for h in bm25.search(question, k=DEPTH)]
        dense_ids = [h.chunk.chunk_id for h in dense.search(query_vectors[position], k=DEPTH)]
        order[query_id] = reciprocal_rank_fusion({"bm25": lexical, "dense": dense_ids})[:DEPTH]

    return Shortlists("scifact", order, benchmark.relevant, benchmark.queries, texts)


MATERIAL = 0.01
"""How large an MRR difference has to be before it counts as an effect.

An earlier version of `verdict` treated any delta above zero as a rescue, and
the first run came back with the capacity control at **+0.0008** — which that
rule would have reported as "more capacity was enough". On 300 queries with
roughly one relevant document each, 0.0008 MRR is about a quarter of one
query moving up a rank. It is noise, and a conclusion that turns on it is a
coin flip wearing a lab coat.

0.01 is chosen as roughly three queries out of 300 moving from rank two to
rank one — small, but a real change rather than a rounding artifact. It is a
judgement rather than a test statistic, which is why it is a named constant
with this note attached instead of a bare number inside an `if`.
"""


def headroom(corpus: dict) -> float:
    """How much a *perfect* reordering of the same shortlist would have been worth.

    The ceiling on MRR for any reranker is hit@5 — if every shortlist that
    contains a relevant document had it at rank one, MRR would equal the share
    of shortlists containing one. Subtracting fusion's own MRR gives what
    reordering had left to win.

    This is the number that turns "the reranker did not help" into something
    interpretable. Without it, a flat result is ambiguous between *the
    reranker is useless here* and *there was nothing left to gain*.
    """
    return round(corpus["baseline"]["hit_at_5"] - corpus["baseline"]["mrr"], 4)


def captured(corpus: dict, label: str) -> float:
    """The share of that headroom a given model actually took."""
    base = corpus["baseline"]["mrr"]
    row = next(r for r in corpus["rerankers"] if r["label"] == label)
    available = headroom(corpus)
    return round((row["scores"]["mrr"] - base) / available, 4) if available else 0.0


def verdict(results: dict) -> str:
    """Read the comparison and say which explanation it supports.

    Written as a function so the conclusion is derived from the numbers rather
    than typed next to them, and so it changes if the numbers do. A test feeds
    it fabricated results and asserts it returns different conclusions for
    them, which is what would catch a hard-coded answer.
    """

    def delta(corpus: str, label: str) -> float:
        base = results[corpus]["baseline"]["mrr"]
        row = next(r for r in results[corpus]["rerankers"] if r["label"] == label)
        return row["scores"]["mrr"] - base

    same_data_small = delta("scifact", "MiniLM-L6")
    same_data_big = delta("scifact", "MiniLM-L12")
    other_data = delta("scifact", "bge-base")

    helped = lambda d: d >= MATERIAL  # noqa: E731
    hurt = lambda d: d <= -MATERIAL  # noqa: E731

    # Checked first: if both help materially, neither explanation is isolated,
    # and reporting one of them would be picking a winner by branch order.
    if helped(other_data) and helped(same_data_big):
        return "mixed: both capacity and training data move the result materially"
    if helped(other_data) and not helped(same_data_big):
        return (
            "supports the domain hypothesis: a reranker trained on broader data "
            "helps where neither MS MARCO model does, and more capacity on the "
            "same training data does not rescue it"
        )
    if helped(same_data_big) and not helped(same_data_small):
        return (
            "contradicts the domain hypothesis: more capacity on the same "
            "training data was enough, so the problem was model size"
        )
    if hurt(same_data_small) and not helped(same_data_big) and not helped(other_data):
        return (
            "no reranker tested helps materially on SciFact. The original model "
            "hurts, the others are indistinguishable from doing nothing, so the "
            "honest reading is that reranking buys nothing here rather than that "
            "a better reranker would"
        )
    return "no material effect in either direction from any model tested"


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args(argv)

    results: dict = {}
    for shortlists in (restaurant_shortlists(), scifact_shortlists(args.limit_docs)):
        baseline = shortlists.baseline
        rows = []
        for reranker in RERANKERS:
            row = rerank_all(shortlists, reranker)
            rows.append(row)
            log.info(
                "reranked",
                extra={
                    "corpus": shortlists.name,
                    "model": reranker.label,
                    "mrr": row["scores"]["mrr"],
                    "delta": round(row["scores"]["mrr"] - baseline["mrr"], 4),
                },
            )
        results[shortlists.name] = {
            "queries": len(shortlists.order),
            "baseline": baseline,
            "rerankers": rows,
        }

    payload = {
        "depth": DEPTH,
        "truncated": args.limit_docs is not None,
        "rerankers": [r.__dict__ for r in RERANKERS],
        "corpora": results,
        "verdict": verdict(results),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "rerankers.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info("artifact written", extra={"path": str(out), "verdict": payload["verdict"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
