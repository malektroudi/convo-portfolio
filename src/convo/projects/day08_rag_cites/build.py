"""Embed the corpus under all three chunkings, plus the question set.

    uv sync --extra train
    uv run python -m convo.projects.day08_rag_cites.build

Writes one artifact the hub reads with numpy alone. The question embeddings
ship with it so dense retrieval works in a container that has no encoder —
the same bargain as days 2, 3 and 5, and the reason the image is 588 MB
rather than over a gigabyte.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from convo.core import get_logger, setup
from convo.projects.day03_embed_router.embed import BASE_MODEL, embed, load_encoder
from convo.projects.day08_rag_cites.chunking import STRATEGIES, chunk_corpus, load_corpus
from convo.projects.day08_rag_cites.questions import QUESTIONS

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
log = get_logger("day08.build")


def main() -> int:
    setup()
    documents = load_corpus()
    log.info(
        "corpus loaded",
        extra={
            "documents": len(documents),
            "characters": sum(len(d.text) for d in documents),
        },
    )

    model, tokenizer = load_encoder(BASE_MODEL)
    started = time.perf_counter()

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {
        "encoder": BASE_MODEL,
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "characters": len(d.text)} for d in documents
        ],
        "strategies": {},
    }

    for strategy in STRATEGIES:
        chunks = chunk_corpus(documents, strategy.name)
        vectors = embed([c.text for c in chunks], model, tokenizer, max_len=256)
        arrays[f"emb_{strategy.name}"] = vectors.astype(np.float32)
        lengths = [len(c.text) for c in chunks]
        meta["strategies"][strategy.name] = {  # type: ignore[index]
            "describe": strategy.describe,
            "chunks": len(chunks),
            "mean_chars": round(sum(lengths) / len(lengths), 1),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "starts_mid_sentence": sum(1 for c in chunks if c.text[:1].islower()),
            "items": [
                {
                    "doc_id": c.doc_id,
                    "start": c.start,
                    "end": c.end,
                    "heading": c.heading,
                    "index": c.index,
                    "strategy": c.strategy,
                }
                for c in chunks
            ],
        }
        log.info(
            "chunked and embedded",
            extra={"strategy": strategy.name, "chunks": len(chunks)},
        )

    question_vectors = embed([q.text for q in QUESTIONS], model, tokenizer, max_len=64)
    arrays["emb_questions"] = question_vectors.astype(np.float32)
    meta["questions"] = [
        {"text": q.text, "expect_doc": q.expect_doc, "probe": q.probe} for q in QUESTIONS
    ]
    meta["built_seconds"] = round(time.perf_counter() - started, 1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARTIFACT_DIR / "index.npz", **arrays)
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = (ARTIFACT_DIR / "index.npz").stat().st_size
    log.info(
        "artifact written",
        extra={
            "path": str(ARTIFACT_DIR / "index.npz"),
            "kilobytes": round(size / 1024),
            "questions": len(QUESTIONS),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
