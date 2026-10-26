"""Was the reranker failing on SciFact, or was it reading half a document?

The corpus-size sweep was built on a premise that turns out to be false.
Fusion scores hit@1 of **0.571** on the restaurant corpus and **0.580** on
SciFact: the two are almost exactly as hard for the retriever, despite a
hundredfold difference in size. Corpus size cannot be what separates them,
and a sweep over it was never going to explain anything.

What does differ is the documents. Under the cross-encoder's own tokenizer:

    restaurant   49 chunks    median   31 tokens    0.0% over 256
    scifact    5183 docs      median  314 tokens   70.4% over 256

Day 9 reranks with `max_length=256`. That is a comfortable fit for chunks
whose median length is 31 tokens and it silently truncates **seven out of ten
SciFact documents**, so for most candidates the model was scoring a fragment —
and the sentence that makes an abstract relevant can easily sit past the cut.

This is not a property of the model, the corpus or the domain. It is a
parameter I chose on day 9 for a corpus where it could not possibly matter,
and then carried unchanged onto one where it does.

So: identical queries, identical shortlists, identical model, one variable.
All 300 queries, which puts the materiality band at 0.01 MRR rather than the
0.036 that made the previous run unreadable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from convo.core import get_logger, setup
from convo.projects.day09_retrieval_evals.build import CROSS_ENCODER

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
WINDOWS = (256, 512)
"""Day 9's setting, and the largest this model accepts."""

log = get_logger("benchmark.truncation")


def token_lengths(texts: list[str]) -> dict:
    """How long these documents are in the reranker's own tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER)
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    ordered = sorted(lengths)
    return {
        "documents": len(lengths),
        "median": round(statistics.median(lengths)),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": max(lengths),
        **{
            f"over_{window}": round(sum(1 for x in lengths if x > window) / len(lengths), 4)
            for window in WINDOWS
        },
    }


def run(windows: tuple[int, ...] = WINDOWS) -> dict:
    from convo.benchmark.rerankers import (
        RERANKERS,
        rerank_all,
        restaurant_shortlists,
        scifact_shortlists,
    )

    reranker = RERANKERS[0]  # day 9's own model
    corpora = {}

    for name, shortlists in (
        ("restaurant", restaurant_shortlists()),
        ("scifact", scifact_shortlists()),
    ):
        baseline = shortlists.baseline
        lengths = token_lengths(list(shortlists.texts.values()))
        rows = []
        for window in windows:
            row = rerank_all(shortlists, reranker, max_length=window)
            row["delta_mrr"] = round(row["scores"]["mrr"] - baseline["mrr"], 4)
            rows.append(row)
            log.info(
                "window measured",
                extra={
                    "corpus": name,
                    "max_length": window,
                    "mrr": row["scores"]["mrr"],
                    "delta": row["delta_mrr"],
                },
            )
        corpora[name] = {
            "queries": len(shortlists.order),
            "baseline": baseline,
            "lengths": lengths,
            "windows": rows,
        }

    return {
        "reranker": reranker.model_id,
        "windows": list(windows),
        "corpora": corpora,
        "verdict": verdict(corpora),
    }


def verdict(corpora: dict) -> str:
    """Does widening the window rescue SciFact? Derived from the numbers."""
    from convo.benchmark.rerankers import material

    sci = corpora["scifact"]
    band = material(sci["queries"])
    rows = {r["max_length"]: r["delta_mrr"] for r in sci["windows"]}
    narrow, wide = min(rows), max(rows)

    improvement = rows[wide] - rows[narrow]
    truncated = sci["lengths"].get(f"over_{narrow}", 0.0)

    if rows[wide] >= band > rows[narrow]:
        return (
            f"truncation explains it: {truncated:.0%} of documents were longer "
            f"than the {narrow}-token window day 9 chose, and widening it to "
            f"{wide} turns a {rows[narrow]:+.4f} MRR change into {rows[wide]:+.4f}. "
            f"The reranker was not failing on this corpus, it was reading part "
            f"of each document"
        )
    if improvement >= band:
        return (
            f"truncation matters but does not fully explain it: widening the "
            f"window moves the result by {improvement:+.4f} MRR, which is "
            f"material, but the reranker still does not beat fusion"
        )
    if abs(improvement) < band:
        return (
            f"truncation is not the explanation: {truncated:.0%} of documents "
            f"exceed the narrow window, yet widening it moves the result by "
            f"only {improvement:+.4f} MRR, inside the {band:.3f} band"
        )
    return f"widening the window makes it worse by {improvement:+.4f} MRR"


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)

    payload = run(tuple(args.windows) if args.windows else WINDOWS)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_DIR / "truncation.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log.info("artifact written", extra={"path": str(out), "verdict": payload["verdict"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
