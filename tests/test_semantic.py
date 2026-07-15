"""Tests for reporadar.semantic — embedding search + hybrid fusion (mocked embeddings)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from reporadar.semantic import semantic_search
from reporadar.store import PaperStore

# A tiny deterministic "embedding": a bag-of-words count vector over a fixed vocab,
# so cosine similarity tracks term overlap — enough to verify the ranking pipeline
# without pulling in sentence-transformers.
_VOCAB = ["lora", "adapter", "diffusion", "image", "retrieval", "graph"]


def _embed(text: str) -> np.ndarray:
    t = text.lower()
    return np.array([float(t.count(w)) for w in _VOCAB], dtype=np.float32) + 1e-3


def _embed_paper(paper: dict) -> np.ndarray:
    return _embed(paper["title"] + " " + paper["abstract"])


def _paper(arxiv_id: str, title: str, abstract: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["A"],
        "abstract": abstract,
        "categories": ["cs.LG"],
        "published": "2024-01-01T00:00:00+00:00",
        "updated": None,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": None,
    }


def _seed(store: PaperStore) -> None:
    store.upsert_papers(
        [
            _paper("p1", "LoRA adapters for fine-tuning", "low-rank adapter modules"),
            _paper("p2", "Diffusion image models", "denoising diffusion for images"),
            _paper("p3", "Graph retrieval", "retrieval over graphs"),
        ]
    )


def _fakes():
    return patch.multiple(
        "reporadar.embeddings",
        compute_paper_embedding=_embed_paper,
        compute_embedding=_embed,
    )


class TestSemanticSearch:
    def test_ranks_semantically_relevant_first(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            with _fakes():
                out = semantic_search(store, "lora adapter", limit=3)
        assert out[0]["arxiv_id"] == "p1"
        assert "search_score" in out[0]

    def test_hybrid_returns_fused_results(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            with _fakes():
                out = semantic_search(store, "diffusion image", limit=3, hybrid=True)
        assert out[0]["arxiv_id"] == "p2"

    def test_respects_limit(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            with _fakes():
                assert len(semantic_search(store, "image", limit=1)) == 1

    def test_blank_query_and_zero_limit(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            with _fakes():
                assert semantic_search(store, "   ", limit=3) == []
                assert semantic_search(store, "image", limit=0) == []

    def test_no_papers(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store, _fakes():
            assert semantic_search(store, "anything", limit=3) == []
