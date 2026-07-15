"""Tests for reporadar.search — full-corpus BM25 search."""

from __future__ import annotations

from reporadar.search import search_corpus


def _paper(arxiv_id: str, title: str, abstract: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "published": "2024-01-01T00:00:00+00:00",
    }


CORPUS = [
    _paper(
        "1",
        "Low-Rank Adaptation of Large Language Models",
        "We propose LoRA, a low-rank adaptation method for efficient fine-tuning.",
    ),
    _paper(
        "2",
        "Retrieval-Augmented Generation",
        "RAG combines retrieval with generation for knowledge-intensive tasks.",
    ),
    _paper(
        "3",
        "Diffusion Models for Image Synthesis",
        "Denoising diffusion probabilistic models generate high-quality images.",
    ),
]


class TestSearchCorpus:
    def test_ranks_relevant_paper_first(self) -> None:
        results = search_corpus(CORPUS, "low-rank adaptation fine-tuning")
        assert results[0]["arxiv_id"] == "1"
        assert results[0]["search_score"] > 0

    def test_adds_search_score_and_preserves_fields(self) -> None:
        top = search_corpus(CORPUS, "diffusion image")[0]
        assert top["arxiv_id"] == "3"
        assert "search_score" in top
        assert top["title"] == "Diffusion Models for Image Synthesis"

    def test_drops_zero_score_papers(self) -> None:
        # Only paper 2 mentions "retrieval"; the others score 0 and are dropped.
        assert [r["arxiv_id"] for r in search_corpus(CORPUS, "retrieval")] == ["2"]

    def test_respects_limit(self) -> None:
        assert len(search_corpus(CORPUS, "models", limit=1)) == 1

    def test_non_positive_limit_returns_empty(self) -> None:
        # A zero/negative limit must not slice with ranked[:limit] (which would
        # silently drop the worst results); it returns [].
        assert search_corpus(CORPUS, "models", limit=0) == []
        assert search_corpus(CORPUS, "models", limit=-3) == []

    def test_blank_query_returns_empty(self) -> None:
        assert search_corpus(CORPUS, "   ") == []
        assert search_corpus(CORPUS, "") == []

    def test_empty_corpus_returns_empty(self) -> None:
        assert search_corpus([], "anything") == []

    def test_does_not_mutate_input(self) -> None:
        before = [dict(p) for p in CORPUS]
        search_corpus(CORPUS, "lora")
        assert before == CORPUS
