"""Tests for reporadar.retrieval (hybrid BM25 + RRF fusion)."""

from __future__ import annotations

from types import SimpleNamespace

from reporadar.retrieval import bm25_ranked_ids, bm25_scores, hybrid_reorder, rrf_fuse


class TestRrfFuse:
    def test_top_of_both_rankings_wins(self) -> None:
        fused = rrf_fuse([["a", "b", "c"], ["a", "c", "b"]])
        assert fused["a"] == 1 / 61 + 1 / 61
        assert fused["a"] > fused["b"]
        assert fused["a"] > fused["c"]

    def test_id_missing_from_one_ranking_still_scores(self) -> None:
        fused = rrf_fuse([["a", "b"], ["b"]])
        assert fused["b"] > fused["a"]  # b: 1/62 + 1/61 ; a: 1/61 only


class TestBm25:
    def test_empty_corpus(self) -> None:
        assert bm25_scores(["x"], []) == []

    def test_relevant_doc_scores_higher(self) -> None:
        corpus = [["retrieval", "index", "vectors"], ["cats", "dogs"]]
        scores = bm25_scores(["retrieval", "vectors"], corpus)
        assert scores[0] > scores[1]
        assert scores[1] == 0.0

    def test_ranked_ids_orders_by_relevance(self) -> None:
        profile = SimpleNamespace(keywords=[("retrieval", 1.0)], anchors=["faiss"], domains=[])
        papers = [
            {"arxiv_id": "off", "title": "Cats", "abstract": "about cats and dogs"},
            {"arxiv_id": "hit", "title": "Retrieval", "abstract": "faiss retrieval index"},
        ]
        assert bm25_ranked_ids(papers, profile) == ["hit", "off"]


class TestHybridReorder:
    def test_surfaces_lexically_strong_but_heuristically_buried(self) -> None:
        profile = SimpleNamespace(
            keywords=[("faiss retrieval", 1.0)],
            anchors=["faiss"],
            domains=["information retrieval"],
        )
        # D is dead last by the heuristic score, but the only real lexical match.
        papers = [
            {"arxiv_id": "A", "title": "A", "abstract": "unrelated cats"},
            {"arxiv_id": "B", "title": "B", "abstract": "unrelated dogs"},
            {"arxiv_id": "C", "title": "C", "abstract": "unrelated birds"},
            {
                "arxiv_id": "D",
                "title": "D",
                "abstract": "faiss retrieval and information retrieval",
            },
        ]
        scores = [
            {"arxiv_id": "A", "score_total": 0.9},
            {"arxiv_id": "B", "score_total": 0.8},
            {"arxiv_id": "C", "score_total": 0.7},
            {"arxiv_id": "D", "score_total": 0.6},
        ]
        out = hybrid_reorder(scores, papers, profile)

        # RRF lifts D (heuristic-last, BM25-first) above B and C; A stays on top.
        assert [s["arxiv_id"] for s in out] == ["A", "D", "B", "C"]
        assert all("rrf_score" in s for s in out)
        # score_total is preserved — only the order changes.
        assert next(s for s in out if s["arxiv_id"] == "D")["score_total"] == 0.6

    def test_no_lexical_signal_keeps_heuristic_order(self) -> None:
        # If nothing matches the query, BM25 is all-zero and ties by original index,
        # so RRF reinforces the heuristic order rather than scrambling it.
        profile = SimpleNamespace(keywords=[("quantum", 1.0)], anchors=[], domains=[])
        papers = [
            {"arxiv_id": "A", "title": "A", "abstract": "cats"},
            {"arxiv_id": "B", "title": "B", "abstract": "dogs"},
        ]
        scores = [{"arxiv_id": "A", "score_total": 0.9}, {"arxiv_id": "B", "score_total": 0.8}]
        out = hybrid_reorder(scores, papers, profile)
        assert [s["arxiv_id"] for s in out] == ["A", "B"]
