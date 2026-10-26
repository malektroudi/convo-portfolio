"""Load a BEIR subset into the exact shapes days 8 and 9 already use.

Every evaluation set in this repository was written by me. That is fine for
showing a failure mode on purpose — day 8's corpus contains a superseded
figure and a near-duplicate policy pair *because* I put them there — and it is
useless for the question a reader actually has, which is whether the retrieval
stack is any good.

Fourteen articles is not a retrieval problem. It is why day 8 reached 100%
dense@3 with nothing left to improve, and why day 9 had to relabel at span
level to have something measurable at all. SciFact is 5,183 documents with
300 queries and relevance judgements made by domain experts who have never
heard of this project, which is the only kind of label that can surprise me.

Nothing here reimplements retrieval. Each BEIR document becomes a single
`Chunk` spanning its own text, and the `BM25` and `DenseIndex` classes from
day 8 are used exactly as they are — the point is to test that code, so
changing it would defeat the exercise.

One deliberate difference from days 8 and 9: no chunking. BEIR's protocol is
document retrieval, and splitting the documents would make the numbers
incomparable with every published baseline for this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

from convo.core import get_logger
from convo.projects.day08_rag_cites.chunking import Chunk, Document

STRATEGY = "document"
"""One chunk per document: BEIR's protocol is document retrieval, and
splitting would make these numbers incomparable with every published
baseline for the dataset."""


def _chunk_id(doc_id: str) -> str:
    """The id day 8's `Chunk` will derive for a whole-document chunk."""
    return f"{doc_id}#{STRATEGY}:0"


SCIFACT = "scifact"
NFCORPUS = "nfcorpus"
DEFAULT = SCIFACT

log = get_logger("benchmark.beir")


@dataclass(frozen=True)
class Benchmark:
    """A public retrieval benchmark, in this project's own vocabulary."""

    name: str
    documents: list[Document]
    chunks: list[Chunk]
    queries: dict[str, str]
    """query_id -> query text, for the queries that have judgements."""
    relevant: dict[str, set[str]]
    """query_id -> the chunk ids a human marked relevant.

    Chunk ids rather than document ids, because that is what the retrievers
    return. Day 8 derives `chunk_id` as `doc_id#strategy:index`, so a
    judgement has to be expressed the same way or every query scores zero and
    the bug looks like a retrieval failure.
    """

    @property
    def judgements(self) -> int:
        return sum(len(v) for v in self.relevant.values())

    @property
    def mean_relevant(self) -> float:
        if not self.relevant:
            return 0.0
        return round(self.judgements / len(self.relevant), 2)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "documents": len(self.documents),
            "queries": len(self.queries),
            "judgements": self.judgements,
            "mean_relevant_per_query": self.mean_relevant,
        }


def load(name: str = DEFAULT, split: str = "test", limit_docs: int | None = None) -> Benchmark:
    """Fetch a BEIR subset and reshape it. Needs the `train` extra for datasets.

    `limit_docs` exists for the tests, which need the shapes rather than the
    corpus. It keeps every judged document and truncates the rest, so a
    shrunken benchmark is still answerable — truncating blindly would silently
    remove the correct answers and make every retriever look broken.
    """
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    raw_queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split=split)

    # BEIR ids are ints in some subsets and strings in others.
    judged_docs = {str(x) for x in qrels["corpus-id"]}
    judged_queries = {str(x) for x in qrels["query-id"]}

    documents: list[Document] = []
    for row in corpus:
        doc_id = str(row["_id"])
        if limit_docs is not None and doc_id not in judged_docs and len(documents) >= limit_docs:
            continue
        title = (row["title"] or "").strip()
        text = (row["text"] or "").strip()
        body = f"{title}\n\n{text}".strip() if title else text
        if not body:
            continue
        documents.append(Document(doc_id=doc_id, title=title or doc_id, text=body))

    queries = {
        str(row["_id"]): row["text"].strip()
        for row in raw_queries
        if str(row["_id"]) in judged_queries and row["text"].strip()
    }

    present = {d.doc_id for d in documents}
    relevant: dict[str, set[str]] = {}
    for row in qrels:
        if int(row["score"]) <= 0:
            continue
        query_id, doc_id = str(row["query-id"]), str(row["corpus-id"])
        if query_id in queries and doc_id in present:
            relevant.setdefault(query_id, set()).add(_chunk_id(doc_id))

    # A query whose judged documents all fell outside the corpus cannot be
    # answered by anybody, and scoring it would be measuring the loader.
    queries = {k: v for k, v in queries.items() if relevant.get(k)}
    relevant = {k: v for k, v in relevant.items() if k in queries}

    chunks = [
        Chunk(
            doc_id=d.doc_id,
            text=d.text,
            start=0,
            end=len(d.text),
            strategy=STRATEGY,
            heading=d.title,
            index=0,
        )
        for d in documents
    ]

    log.info(
        "benchmark loaded",
        extra={
            "benchmark": name,
            "documents": len(documents),
            "queries": len(queries),
            "judgements": sum(len(v) for v in relevant.values()),
        },
    )
    return Benchmark(name, documents, chunks, queries, relevant)
