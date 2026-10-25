"""Span-level citation: narrow a retrieved chunk to the sentences that answer.

A chunk is 150–400 characters. Citing the whole thing and calling it a source
is the thing most RAG demos do, and it is nearly useless to a reader — they
still have to find the claim themselves, and nothing stops the chunk being
merely *about* the topic rather than containing the answer.

So retrieval picks the chunk and this picks the sentences inside it, carrying
offsets the whole way so the final citation is an exact span of the original
document. The same invariant as chunking applies and is tested:

    document.text[citation.start:citation.end] == citation.text

Sentence selection is term overlap rather than a second embedding pass. That is
a deliberate trade: it is deterministic, free, testable, and explainable to a
reader — and it keeps the hub model-free. Where it fails is a sentence that
answers in different words from the question, which is exactly the case day 9's
reranker exists to fix.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from convo.projects.day08_rag_cites.chunking import Chunk, Document
from convo.projects.day08_rag_cites.index import tokenise

# Split after . ! ? when followed by space/newline, keeping offsets recoverable.
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    """Absolute offset into the source document."""
    end: int


@dataclass(frozen=True)
class Citation:
    """An exact span of a document, with the evidence for choosing it."""

    doc_id: str
    text: str
    start: int
    end: int
    score: float
    chunk_id: str
    heading: str = ""
    matched_terms: tuple[str, ...] = ()

    def verify(self, document: Document) -> bool:
        return document.span(self.start, self.end) == self.text

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "heading": self.heading,
            "matched_terms": list(self.matched_terms),
        }


def sentences(chunk: Chunk) -> list[Sentence]:
    """Split a chunk into sentences carrying absolute document offsets."""
    out: list[Sentence] = []
    cursor = 0
    for piece in SENTENCE.split(chunk.text):
        if not piece.strip():
            cursor += len(piece)
            continue
        found = chunk.text.find(piece, cursor)
        if found == -1:  # pragma: no cover - split always yields substrings
            continue
        stripped = piece.strip()
        lead = len(piece) - len(piece.lstrip())
        start = chunk.start + found + lead
        out.append(Sentence(stripped, start, start + len(stripped)))
        cursor = found + len(piece)
    return out


def _overlap(sentence: str, query_terms: set[str]) -> tuple[float, tuple[str, ...]]:
    terms = tokenise(sentence)
    if not terms:
        return 0.0, ()
    matched = tuple(sorted(set(terms) & query_terms))
    if not matched:
        return 0.0, ()
    # Coverage of the question, with a mild length penalty so a long sentence
    # does not win merely by containing more words.
    coverage = len(matched) / max(len(query_terms), 1)
    density = len(matched) / math.sqrt(len(terms))
    return round(coverage * 0.75 + density * 0.25, 6), matched


def cite(chunk: Chunk, query: str, max_sentences: int = 2) -> Citation | None:
    """The best contiguous run of sentences in `chunk` for `query`.

    Contiguous on purpose: an answer split across two adjacent sentences
    ("We hold your table for fifteen minutes. After that we may release it.")
    should cite both, while two unrelated sentences from opposite ends of a
    chunk should not be stitched into one misleading span.
    """
    query_terms = set(tokenise(query))
    pieces = sentences(chunk)
    if not pieces or not query_terms:
        return None

    best_score = 0.0
    best: tuple[int, int] | None = None
    best_terms: tuple[str, ...] = ()

    for start_index in range(len(pieces)):
        for length in range(1, max_sentences + 1):
            end_index = start_index + length
            if end_index > len(pieces):
                break
            window = pieces[start_index:end_index]
            joined = " ".join(s.text for s in window)
            score, matched = _overlap(joined, query_terms)
            # A longer window must earn its extra text rather than win by volume.
            score -= 0.02 * (length - 1)
            if score > best_score:
                best_score, best, best_terms = score, (start_index, end_index), matched

    if best is None:
        return None

    window = pieces[best[0] : best[1]]
    return Citation(
        doc_id=chunk.doc_id,
        text="",  # filled by cite_in_document, which has the source text
        start=window[0].start,
        end=window[-1].end,
        score=best_score,
        chunk_id=chunk.chunk_id,
        heading=chunk.heading,
        matched_terms=best_terms,
    )


def cite_in_document(
    chunk: Chunk, query: str, document: Document, max_sentences: int = 2
) -> Citation | None:
    """`cite`, with the span's text read back from the document itself.

    Reading the text from the source rather than carrying it along is what
    makes the citation verifiable: if the offsets were wrong, the text would be
    wrong too, and the invariant test would catch it.
    """
    bare = cite(chunk, query, max_sentences)
    if bare is None:
        return None
    return Citation(
        doc_id=bare.doc_id,
        text=document.span(bare.start, bare.end),
        start=bare.start,
        end=bare.end,
        score=bare.score,
        chunk_id=bare.chunk_id,
        heading=bare.heading,
        matched_terms=bare.matched_terms,
    )
