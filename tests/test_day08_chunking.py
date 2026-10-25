"""Chunking tests, built around the one invariant that makes citation real.

    document.text[chunk.start:chunk.end] == chunk.text

If that ever fails, every citation in the system is a guess. It is asserted for
every chunk of every document under every strategy, which is cheap and catches
the entire class of off-by-one that offset arithmetic invites.
"""

from __future__ import annotations

import pytest

from convo.projects.day08_rag_cites.chunking import (
    STRATEGIES,
    Document,
    chunk_corpus,
    fixed,
    load_corpus,
    structural,
)

SAMPLE = Document(
    doc_id="sample",
    title="Sample",
    text=(
        "# Title\n\n"
        "## First section\n\n"
        "The first paragraph says something short.\n\n"
        "## Second section\n\n"
        "The second paragraph says something else entirely.\n"
    ),
)


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


class TestCorpus:
    def test_the_corpus_loads(self, corpus):
        assert len(corpus) >= 12
        assert all(d.text.strip() for d in corpus)

    def test_every_document_has_a_title_from_its_heading(self, corpus):
        assert all(d.title and not d.title.startswith("#") for d in corpus)
        titles = {d.doc_id: d.title for d in corpus}
        assert titles["parking"] == "Parking and getting here"

    def test_line_endings_are_normalised(self, corpus):
        # Offsets are indexes into this text; a stray CR shifts all of them.
        assert all("\r" not in d.text for d in corpus)

    def test_the_corpus_contains_its_designed_failure_modes(self, corpus):
        ids = {d.doc_id for d in corpus}
        assert {"cancellation-standard", "cancellation-private-events"} <= ids, (
            "near-duplicate pair"
        )
        assert {"opening-hours", "holiday-hours"} <= ids, "split information"
        assert "deposit-policy-2026" in ids, "superseded policy"

        everything = " ".join(d.text.lower() for d in corpus)
        assert "ev charging" not in everything, "the unanswerable question must stay unanswerable"


class TestTheInvariant:
    @pytest.mark.parametrize("strategy", [s.name for s in STRATEGIES])
    def test_every_chunk_slices_back_to_itself(self, corpus, strategy):
        chunks = chunk_corpus(corpus, strategy)
        by_id = {d.doc_id: d for d in corpus}
        assert chunks, strategy
        for chunk in chunks:
            document = by_id[chunk.doc_id]
            assert chunk.verify(document), (
                f"{chunk.chunk_id}: offsets {chunk.start}:{chunk.end} do not "
                f"slice back to the chunk text"
            )

    @pytest.mark.parametrize("strategy", [s.name for s in STRATEGIES])
    def test_offsets_are_ordered_and_in_range(self, corpus, strategy):
        by_id = {d.doc_id: d for d in corpus}
        for chunk in chunk_corpus(corpus, strategy):
            assert 0 <= chunk.start < chunk.end <= len(by_id[chunk.doc_id].text)

    @pytest.mark.parametrize("strategy", [s.name for s in STRATEGIES])
    def test_no_chunk_is_empty_or_pure_whitespace(self, corpus, strategy):
        assert all(c.text.strip() for c in chunk_corpus(corpus, strategy))

    @pytest.mark.parametrize("strategy", [s.name for s in STRATEGIES])
    def test_chunk_ids_are_unique(self, corpus, strategy):
        chunks = chunk_corpus(corpus, strategy)
        assert len({c.chunk_id for c in chunks}) == len(chunks)


class TestFixed:
    def test_it_cuts_at_the_requested_size(self):
        chunks = fixed(SAMPLE, size=50, overlap=0)
        assert all(len(c.text) <= 50 for c in chunks)
        assert len(chunks) > 1

    def test_without_overlap_chunks_do_not_share_text(self):
        chunks = fixed(SAMPLE, size=50, overlap=0)
        spans = [(c.start, c.end) for c in chunks]
        for (_, end), (start, _) in zip(spans[:-1], spans[1:], strict=True):
            assert start >= end, "non-overlapping chunks must not overlap"

    def test_with_overlap_neighbours_share_text(self):
        chunks = fixed(SAMPLE, size=50, overlap=20)
        assert len(chunks) > 1
        shared = [
            SAMPLE.text[b.start : a.end]
            for a, b in zip(chunks[:-1], chunks[1:], strict=True)
            if b.start < a.end
        ]
        assert shared, "overlap should make at least one neighbour pair share text"

    def test_overlap_is_recorded_in_the_strategy_name(self):
        assert fixed(SAMPLE, size=50, overlap=20)[0].strategy == "fixed-overlap-20"

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (50, 50), (50, 60)])
    def test_nonsense_parameters_are_refused(self, size, overlap):
        with pytest.raises(ValueError):
            fixed(SAMPLE, size=size, overlap=overlap)

    def test_the_whole_document_is_covered_without_overlap(self):
        chunks = fixed(SAMPLE, size=50, overlap=0)
        # Concatenating the spans back, ignoring the whitespace trimmed at the
        # edges, must reproduce the document's non-whitespace content.
        joined = "".join(SAMPLE.text[c.start : c.end] for c in chunks)
        assert "".join(joined.split()) == "".join(SAMPLE.text.split())


class TestStructural:
    def test_it_splits_on_headings(self):
        chunks = structural(SAMPLE)
        headings = [c.heading for c in chunks if c.heading]
        assert "First section" in headings
        assert "Second section" in headings

    def test_the_heading_is_metadata_not_part_of_the_cited_span(self):
        chunk = next(c for c in structural(SAMPLE) if c.heading == "First section")
        assert chunk.heading == "First section"
        assert "## First section" not in chunk.text, (
            "the heading must not be inside the span a citation points at"
        )
        assert chunk.text.startswith("The first paragraph")

    def test_a_document_with_no_headings_still_chunks(self):
        plain = Document("plain", "Plain", "Just one paragraph with no headings at all.\n")
        chunks = structural(plain)
        assert len(chunks) == 1
        assert chunks[0].verify(plain)

    def test_an_over_long_section_is_split_on_paragraphs(self):
        body = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(20))
        document = Document("long", "Long", f"# Long\n\n## Section\n\n{body}\n")
        chunks = structural(document, max_size=200)
        assert len(chunks) > 1
        assert all(c.verify(document) for c in chunks)

    def test_structural_chunks_rarely_start_mid_sentence(self, corpus):
        chunks = chunk_corpus(corpus, "structural")
        bad = [c for c in chunks if c.text[:1].islower()]
        assert len(bad) / len(chunks) < 0.1, (
            f"{len(bad)} of {len(chunks)} structural chunks start mid-sentence"
        )


class TestStrategiesDiffer:
    def test_the_three_strategies_produce_different_chunkings(self, corpus):
        counts = {s.name: len(chunk_corpus(corpus, s.name)) for s in STRATEGIES}
        assert len(set(counts.values())) > 1, f"strategies are indistinguishable: {counts}"

    def test_fixed_cuts_more_sentences_than_structural(self, corpus):
        def mid_sentence(strategy: str) -> int:
            return sum(1 for c in chunk_corpus(corpus, strategy) if c.text[:1].islower())

        assert mid_sentence("fixed") > mid_sentence("structural"), (
            "ignoring structure should cost something measurable"
        )

    def test_an_unknown_strategy_is_refused(self, corpus):
        with pytest.raises(ValueError, match="unknown strategy"):
            chunk_corpus(corpus, "vibes")
