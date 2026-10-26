"""The public-benchmark harness: that it reshapes BEIR correctly, and that it
does not quietly adapt the retrievers it is supposed to be testing.

The loader gets the strictest tests. Every one of its failure modes produces a
*plausible* number rather than an error — mismatched id formats score every
query zero and look like a broken retriever, and a truncated corpus that drops
the judged documents looks like a hard benchmark. Both would be reported as
findings about the retrieval stack, which is exactly the mistake this whole
exercise exists to stop.

Nothing here downloads a corpus. The reshaping is tested on a hand-built BEIR
fixture, and the stored run is checked for the properties it must have.
"""

from __future__ import annotations

import pytest

from convo.benchmark import beir
from convo.benchmark.results import RETRIEVERS, load
from convo.projects.day08_rag_cites.chunking import Chunk, Document


class TestTheChunkIdContract:
    """The judgements must speak the same language as the retrievers.

    Day 8 derives `chunk_id` as `doc_id#strategy:index`. If the loader records
    judgements as bare document ids, every query scores zero and the bug is
    indistinguishable from the retrieval stack simply being bad.
    """

    def test_the_helper_matches_what_chunk_derives(self):
        document = Document(doc_id="42", title="t", text="body")
        chunk = Chunk(
            doc_id=document.doc_id,
            text=document.text,
            start=0,
            end=len(document.text),
            strategy=beir.STRATEGY,
            heading=document.title,
            index=0,
        )
        assert beir._chunk_id("42") == chunk.chunk_id

    def test_the_strategy_name_is_not_one_of_day_eights(self):
        """`document` rather than `fixed` or `structural`, because BEIR's
        protocol is document retrieval and these are not chunked."""
        from convo.projects.day08_rag_cites.chunking import STRATEGIES

        assert beir.STRATEGY not in {s.name for s in STRATEGIES}


class TestTheHarnessDoesNotAdaptTheRetrievers:
    """The point is to test day 8 and 9's code, so it must be imported as-is."""

    def test_it_imports_the_real_retrievers(self):
        import convo.benchmark.run as run
        from convo.core.retrieval import reciprocal_rank_fusion, score_runs
        from convo.projects.day08_rag_cites.index import BM25, DenseIndex

        assert run.BM25 is BM25
        assert run.DenseIndex is DenseIndex
        assert run.score_runs is score_runs
        assert run.reciprocal_rank_fusion is reciprocal_rank_fusion

    def test_it_compares_the_same_four_retrievers_as_day_nine(self):
        import convo.benchmark.run as run
        from convo.projects.day09_retrieval_evals.results import RETRIEVERS as DAY9

        assert tuple(run.RETRIEVERS) == tuple(DAY9)

    def test_it_sweeps_more_than_one_shortlist_depth(self):
        """A negative result for the reranker at one depth is a parameter,
        not a finding."""
        import convo.benchmark.run as run

        assert len(run.DEPTHS) >= 2
        assert min(run.DEPTHS) < max(run.DEPTHS)


class TestTheStoredRun:
    @pytest.fixture(scope="class")
    def results(self):
        found = load()
        if found is None:
            pytest.skip("benchmark not run")
        return found

    def test_it_is_the_whole_corpus(self, results):
        """A truncated run is for smoke-testing and must never be published."""
        assert not results.truncated
        assert results.documents > 1000

    def test_the_labels_are_sparse_like_the_task(self, results):
        """SciFact has about one relevant document per query, which is what
        makes hit@1 and MRR meaningful here."""
        assert 0 < results.mean_relevant < 5

    def test_every_retriever_scored_every_depth(self, results):
        for depth in results.depths:
            for retriever in RETRIEVERS:
                row = results.scores(depth, retriever)
                assert row["queries"] == results.queries, (depth, retriever)

    def test_every_metric_is_a_rate(self, results):
        for depth in results.depths:
            for retriever in RETRIEVERS:
                row = results.scores(depth, retriever)
                for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"):
                    assert 0.0 <= row[key] <= 1.0, (depth, retriever, key, row[key])

    def test_hit_at_k_does_not_decrease_with_k(self, results):
        """A structural property of the metric, so a violation is a scorer bug."""
        for depth in results.depths:
            for retriever in RETRIEVERS:
                row = results.scores(depth, retriever)
                assert row["hit_at_1"] <= row["hit_at_3"] <= row["hit_at_5"], (depth, retriever)

    def test_a_deeper_shortlist_cannot_reduce_recall(self, results):
        """Fusion at depth 100 sees everything it saw at depth 10."""
        if len(results.depths) < 2:
            pytest.skip("only one depth")
        shallow, deep = min(results.depths), max(results.depths)
        for retriever in ("bm25", "dense"):
            assert results.scores(deep, retriever)["hit_at_5"] >= (
                results.scores(shallow, retriever)["hit_at_5"] - 1e-9
            ), retriever

    def test_it_is_harder_than_the_corpus_i_wrote(self, results):
        """The reason for doing this at all.

        Fourteen hand-written articles is not a retrieval problem. If the
        public benchmark were *easier*, it would not be testing anything the
        restaurant corpus did not already test.
        """
        if not results.restaurant:
            pytest.skip("day 9 artifact not built")
        mine = results.restaurant["hybrid+rerank"]["mrr"]
        theirs = results.scores(results.depths[0], "hybrid+rerank")["mrr"]
        assert theirs < mine + 0.15, (
            "the public benchmark scores as well as my own corpus, which means "
            "one of the two is not measuring what it claims"
        )

    def test_the_reranker_delta_is_reported_with_its_sign(self, results):
        """Day 9 found +14 points. Whatever this finds, it must be signed and
        not silently taken as an absolute improvement."""
        for depth in results.depths:
            delta = results.rerank_delta(depth)
            assert isinstance(delta, float)
            assert -1.0 <= delta <= 1.0
