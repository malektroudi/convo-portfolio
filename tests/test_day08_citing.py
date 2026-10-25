"""Retrieval and span-level citation over the real corpus.

The invariant carries all the way down: a citation's offsets must slice the
original document back to the citation's own text. Chunking preserved it from
document to chunk; this asserts it from chunk to sentence, which is where the
arithmetic is fiddliest and where a quiet off-by-one would make every
highlighted span subtly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from convo.projects.day08_rag_cites.answer import cite, cite_in_document, sentences
from convo.projects.day08_rag_cites.chunking import chunk_corpus, load_corpus
from convo.projects.day08_rag_cites.index import BM25, DenseIndex, tokenise


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(corpus):
    return chunk_corpus(corpus, "structural")


@pytest.fixture(scope="module")
def by_id(corpus):
    return {d.doc_id: d for d in corpus}


@pytest.fixture(scope="module")
def bm25(chunks):
    return BM25(chunks)


class TestTokenise:
    def test_it_lowercases_and_drops_stopwords(self):
        assert tokenise("How much is THE deposit?") == ["much", "deposit"]

    def test_punctuation_and_empties_vanish(self):
        assert tokenise("...!?") == []

    def test_numbers_survive(self):
        assert "200" in tokenise("a deposit of 200 CAD")


class TestBM25:
    def test_it_finds_the_obvious_article(self, bm25):
        hits = bm25.search("what time do you open", k=3)
        assert hits
        assert any(h.chunk.doc_id in {"opening-hours", "holiday-hours"} for h in hits)

    def test_scores_descend_and_ranks_count_from_one(self, bm25):
        hits = bm25.search("vegan menu", k=5)
        assert hits[0].rank == 1
        assert all(a.score >= b.score for a, b in zip(hits[:-1], hits[1:], strict=True))

    def test_a_query_of_only_stopwords_returns_nothing(self, bm25):
        assert bm25.search("is the a of", k=5) == []

    def test_a_query_matching_nothing_returns_nothing(self, bm25):
        assert bm25.search("helicopter zebra quantum", k=5) == []

    def test_it_distinguishes_the_near_duplicate_policies(self, bm25):
        private = bm25.search("cancel a private event deposit refund", k=3)
        assert private[0].chunk.doc_id == "cancellation-private-events", (
            "the near-duplicate pair is the trap this corpus was built with"
        )

    def test_k_caps_the_results(self, bm25):
        assert len(bm25.search("booking table", k=2)) <= 2


class TestDenseIndex:
    def _index(self, chunks):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(len(chunks), 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return DenseIndex(chunks, vectors), vectors

    def test_a_query_equal_to_a_chunk_vector_retrieves_that_chunk(self, chunks):
        index, vectors = self._index(chunks)
        target = 7
        hits = index.search(vectors[target], k=1)
        assert hits[0].chunk.chunk_id == chunks[target].chunk_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_an_unnormalised_query_is_normalised(self, chunks):
        index, vectors = self._index(chunks)
        hits = index.search(vectors[3] * 17.0, k=1)
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_a_length_mismatch_is_refused(self, chunks):
        with pytest.raises(ValueError, match="embeddings"):
            DenseIndex(chunks, np.zeros((len(chunks) - 1, 8), dtype=np.float32))

    def test_results_are_ranked(self, chunks):
        index, vectors = self._index(chunks)
        hits = index.search(vectors[0], k=5)
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
        assert all(a.score >= b.score for a, b in zip(hits[:-1], hits[1:], strict=True))


class TestSentences:
    def test_offsets_slice_back_to_the_sentence(self, chunks, by_id):
        for chunk in chunks:
            document = by_id[chunk.doc_id]
            for sentence in sentences(chunk):
                assert document.span(sentence.start, sentence.end) == sentence.text

    def test_sentences_stay_inside_their_chunk(self, chunks):
        for chunk in chunks:
            for sentence in sentences(chunk):
                assert chunk.start <= sentence.start < sentence.end <= chunk.end

    def test_a_multi_sentence_chunk_splits(self, chunks):
        multi = next(c for c in chunks if c.text.count(". ") >= 1)
        assert len(sentences(multi)) >= 2


class TestCitation:
    def test_every_citation_slices_back_to_the_document(self, chunks, by_id):
        """The invariant, over every chunk and a spread of real questions."""
        # One question per article theme, so the invariant is exercised across
        # every document rather than wherever six questions happen to land.
        questions = [
            "how do I book a table",
            "how long before can I cancel",
            "cancel a private event and get the deposit back",
            "how much is the deposit",
            "we are a group of twenty people",
            "what time do you open",
            "are you open on christmas day",
            "I have a nut allergy",
            "do you have vegan food",
            "is it wheelchair accessible",
            "where can I park",
            "we are running late",
            "is there a dress code",
            "can we split the bill",
        ]
        checked = 0
        for chunk in chunks:
            document = by_id[chunk.doc_id]
            for question in questions:
                citation = cite_in_document(chunk, question, document)
                if citation is None:
                    continue
                assert citation.verify(document), (
                    f"{citation.chunk_id} {citation.start}:{citation.end} does not slice back"
                )
                checked += 1
        assert checked > 50, "the invariant should have been exercised widely"

    def test_a_citation_is_narrower_than_its_chunk(self, chunks, by_id):
        chunk = next(c for c in chunks if c.doc_id == "late-arrival" and len(c.text) > 120)
        citation = cite_in_document(chunk, "how long do you hold the table", by_id[chunk.doc_id])
        assert citation is not None
        assert len(citation.text) < len(chunk.text), (
            "citing the whole chunk is what day 8 exists to avoid"
        )

    def test_it_picks_the_sentence_that_answers(self, chunks, by_id):
        chunk = next(
            c for c in chunks if c.doc_id == "late-arrival" and "fifteen minutes" in c.text
        )
        citation = cite_in_document(chunk, "how long do you hold my table", by_id[chunk.doc_id])
        assert citation is not None
        assert "fifteen minutes" in citation.text

    def test_matched_terms_are_reported(self, chunks, by_id):
        chunk = next(c for c in chunks if c.doc_id == "parking")
        citation = cite_in_document(chunk, "street parking on dundas", by_id[chunk.doc_id])
        assert citation is not None
        assert "parking" in citation.matched_terms

    def test_an_unrelated_query_cites_nothing(self, chunks):
        chunk = next(c for c in chunks if c.doc_id == "parking")
        assert cite(chunk, "helicopter zebra quantum") is None

    def test_an_empty_query_cites_nothing(self, chunks):
        assert cite(chunks[0], "") is None

    def test_a_two_sentence_answer_can_be_cited_contiguously(self, chunks, by_id):
        chunk = next(c for c in chunks if c.doc_id == "late-arrival" and "release" in c.text)
        citation = cite_in_document(
            chunk,
            "how long do you hold the table before releasing it",
            by_id[chunk.doc_id],
            max_sentences=2,
        )
        assert citation is not None
        assert citation.verify(by_id[chunk.doc_id])

    def test_max_sentences_bounds_the_span(self, chunks, by_id):
        chunk = max((c for c in chunks if c.doc_id == "allergens"), key=lambda c: len(c.text))
        one = cite_in_document(chunk, "allergy kitchen nuts gluten", by_id[chunk.doc_id], 1)
        three = cite_in_document(chunk, "allergy kitchen nuts gluten", by_id[chunk.doc_id], 3)
        assert one is not None and three is not None
        assert len(one.text) <= len(three.text)


class TestEndToEnd:
    def test_retrieve_then_cite_lands_on_the_right_document(self, bm25, by_id):
        question = "how long do you hold my table if I am late"
        hits = bm25.search(question, k=3)
        assert hits
        citation = cite_in_document(hits[0].chunk, question, by_id[hits[0].chunk.doc_id])
        assert citation is not None
        assert citation.doc_id == "late-arrival"
        assert citation.verify(by_id[citation.doc_id])

    def test_the_unanswerable_question_retrieves_weakly(self, bm25):
        # Nothing in the corpus mentions EV charging. BM25 may still return the
        # parking article on the word "charging"; what matters for day 10 is
        # that no sentence in it actually answers.
        hits = bm25.search("do you have EV charging", k=3)
        for hit in hits:
            citation = cite(hit.chunk, "do you have EV charging")
            if citation is not None:
                assert "ev" not in citation.matched_terms
