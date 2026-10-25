"""Gold labels and the chunk-relevance derivation.

The derivation is the load-bearing idea of day 9: label a span once, and chunk
relevance falls out for any chunking, including ones not written yet. If it
were wrong, every number in the comparison would be wrong in the same direction
and nothing would look suspicious.
"""

from __future__ import annotations

import pytest

from convo.projects.day08_rag_cites.chunking import (
    STRATEGIES,
    Chunk,
    Document,
    chunk_corpus,
    load_corpus,
)
from convo.projects.day09_retrieval_evals.gold import (
    ALL,
    GOLD,
    HARD,
    Gold,
    Span,
    covers,
    locate,
    relevant,
)

DOC = Document(
    doc_id="d",
    title="D",
    text="# D\n\nThe cat sat on the mat.\nIt was a\nvery comfortable mat indeed.\n",
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def by_id(corpus):
    return {d.doc_id: d for d in corpus}


class TestLocate:
    def test_it_finds_a_quote_and_returns_exact_offsets(self):
        span = locate(Gold("q", "d", "The cat sat on the mat."), DOC)
        assert DOC.text[span.start : span.end] == "The cat sat on the mat."

    def test_it_matches_across_a_line_wrap(self):
        # The corpus is hard-wrapped, so a labelled sentence usually contains a
        # newline nobody would type into a label.
        span = locate(Gold("q", "d", "It was a very comfortable mat indeed."), DOC)
        assert "\n" in DOC.text[span.start : span.end]
        assert DOC.text[span.start : span.end].split() == [
            "It",
            "was",
            "a",
            "very",
            "comfortable",
            "mat",
            "indeed.",
        ]

    def test_a_missing_quote_is_an_error_not_a_silent_zero(self):
        with pytest.raises(ValueError, match="not found"):
            locate(Gold("q", "d", "the dog sat on the mat"), DOC)

    def test_an_ambiguous_quote_is_refused(self):
        doubled = Document("d", "D", "the mat. and again the mat. end")
        with pytest.raises(ValueError, match="appears 2 times"):
            locate(Gold("q", "d", "the mat."), doubled)


class TestCovers:
    def _chunk(self, start: int, end: int) -> Chunk:
        return Chunk("d", DOC.text[start:end], start, end, "test")

    def test_a_chunk_containing_the_whole_span_covers_it(self):
        span = Span("d", 10, 20)
        assert covers(self._chunk(0, 40), span)

    def test_a_chunk_with_most_of_the_span_still_covers_it(self):
        # Chunk boundaries clip answers; demanding total containment would
        # punish every chunker for the same unavoidable edge.
        span = Span("d", 10, 20)
        assert covers(self._chunk(0, 16), span), "6 of 10 characters is enough"

    def test_a_chunk_with_a_sliver_does_not(self):
        span = Span("d", 10, 20)
        assert not covers(self._chunk(0, 12), span), "2 of 10 characters is not"

    def test_a_chunk_from_another_document_never_covers(self):
        other = Chunk("elsewhere", "text", 10, 20, "test")
        assert not covers(other, Span("d", 10, 20))

    def test_no_overlap_means_no_coverage(self):
        assert not covers(self._chunk(0, 5), Span("d", 10, 20))

    def test_the_threshold_is_adjustable(self):
        span = Span("d", 10, 20)
        chunk = self._chunk(0, 16)
        assert covers(chunk, span, threshold=0.5)
        assert not covers(chunk, span, threshold=0.9)


class TestTheLabelSet:
    def test_every_label_locates_exactly_once(self, by_id):
        for gold in ALL:
            locate(gold, by_id[gold.doc_id])  # raises if missing or ambiguous

    def test_questions_are_unique(self):
        questions = [g.question for g in ALL]
        assert len(questions) == len(set(questions))

    def test_there_are_both_easy_and_hard_questions(self):
        assert len(GOLD) >= 30
        assert len(HARD) >= 8
        assert set(GOLD).isdisjoint(HARD)

    def test_hard_questions_share_little_vocabulary_with_their_answer(self):
        """The point of the hard set: lexical retrieval has nothing to grip."""
        from convo.projects.day08_rag_cites.index import tokenise

        weak = 0
        for gold in HARD:
            overlap = set(tokenise(gold.question)) & set(tokenise(gold.quote))
            if len(overlap) <= 1:
                weak += 1
        assert weak >= len(HARD) * 0.7, (
            "most hard questions should barely overlap their answer's wording"
        )

    @pytest.mark.parametrize("strategy", [s.name for s in STRATEGIES])
    def test_every_question_has_at_least_one_relevant_chunk(self, corpus, by_id, strategy):
        chunks = chunk_corpus(corpus, strategy)
        for gold in ALL:
            span = locate(gold, by_id[gold.doc_id])
            assert relevant(chunks, span), (
                f"{strategy}: {gold.question!r} has no relevant chunk, so any "
                f"score for it would be meaningless"
            )

    @pytest.mark.parametrize("strategy", [s.name for s in STRATEGIES])
    def test_relevant_chunks_really_contain_the_answer(self, corpus, by_id, strategy):
        chunks = chunk_corpus(corpus, strategy)
        by_cid = {c.chunk_id: c for c in chunks}
        for gold in ALL:
            span = locate(gold, by_id[gold.doc_id])
            for chunk_id in relevant(chunks, span):
                chunk = by_cid[chunk_id]
                assert chunk.doc_id == gold.doc_id
                overlap = min(chunk.end, span.end) - max(chunk.start, span.start)
                assert overlap > 0

    def test_a_finer_chunking_does_not_lose_labels(self, corpus, by_id):
        """One label set, every chunking — that is the whole design."""
        counts = {}
        for strategy in STRATEGIES:
            chunks = chunk_corpus(corpus, strategy.name)
            counts[strategy.name] = sum(
                bool(relevant(chunks, locate(g, by_id[g.doc_id]))) for g in ALL
            )
        assert set(counts.values()) == {len(ALL)}, counts
