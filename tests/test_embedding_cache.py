"""Tests for reporadar.embedding_cache — the persistent per-paper embedding cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from reporadar.embedding_cache import (
    default_model,
    embed_papers_cached,
    load_cached_vectors,
    vec_from_bytes,
    vec_to_bytes,
)
from reporadar.store import PaperStore


def _paper(arxiv_id: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": ["A"],
        "abstract": "abstract " + arxiv_id,
        "categories": ["cs.LG"],
        "published": "2024-01-01T00:00:00+00:00",
        "updated": None,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": None,
    }


def _fake_paper_emb(paper: dict) -> np.ndarray:
    """Deterministic vector from the id, so cache hits/misses are verifiable."""
    h = sum(ord(c) for c in paper["arxiv_id"])
    return np.array([h % 7, h % 5, h % 3], dtype=np.float32) + 1.0


class TestVecSerialization:
    def test_round_trip(self) -> None:
        v = np.array([0.5, -1.0, 2.25], dtype=np.float32)
        assert np.allclose(vec_from_bytes(vec_to_bytes(v)), v)


class TestEmbedPapersCached:
    def test_computes_only_uncached_then_reuses(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_papers([_paper("a"), _paper("b")])
            with patch(
                "reporadar.embeddings.compute_paper_embedding", side_effect=_fake_paper_emb
            ) as m:
                first = embed_papers_cached(store, [_paper("a"), _paper("b")])
                assert m.call_count == 2  # both encoded on the first pass
                assert set(first) == {"a", "b"}
                assert store.embedding_count(default_model()) == 2

                m.reset_mock()
                second = embed_papers_cached(store, [_paper("a"), _paper("b")])
                assert m.call_count == 0  # all cached → nothing re-encoded
                assert np.allclose(second["a"], first["a"])

    def test_new_paper_encodes_incrementally(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_papers([_paper("a"), _paper("b"), _paper("c")])
            with patch(
                "reporadar.embeddings.compute_paper_embedding", side_effect=_fake_paper_emb
            ) as m:
                embed_papers_cached(store, [_paper("a")])
                assert m.call_count == 1
                m.reset_mock()
                out = embed_papers_cached(store, [_paper("a"), _paper("b"), _paper("c")])
                assert m.call_count == 2  # only b and c are new
                assert set(out) == {"a", "b", "c"}

    def test_corrupt_blob_is_reencoded(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_paper("a"))
            # A corrupt (non-float32-length) blob under the current model key.
            store.save_paper_embedding("a", default_model(), 3, b"\x00\x01\x02")
            with patch(
                "reporadar.embeddings.compute_paper_embedding", side_effect=_fake_paper_emb
            ) as m:
                out = embed_papers_cached(store, [_paper("a")])
            assert m.call_count == 1  # corrupt blob → treated as uncached, re-encoded
            assert out["a"].shape == (3,)

    def test_default_model_carries_recipe_tag(self) -> None:
        # A recipe suffix lets a future encoding change invalidate stale vectors.
        assert ":" in default_model()

    def test_load_cached_vectors(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_paper("a"))
            with patch("reporadar.embeddings.compute_paper_embedding", side_effect=_fake_paper_emb):
                embed_papers_cached(store, [_paper("a")])
            loaded = load_cached_vectors(store)
            assert "a" in loaded and loaded["a"].shape == (3,)
