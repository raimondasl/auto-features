"""Tests for the seeded-preference eval split logic (evals/seeded.py)."""

from __future__ import annotations

from pathlib import Path

from seeded import (
    build_seeded_store,
    candidate_pool,
    heldout_labels,
    is_relevant,
    split_relevant,
)

from reporadar.store import PaperStore


def _pool(n_gold: int = 5, n_distractor: int = 3, queries: int = 1) -> list[dict]:
    pool = [
        {
            "arxiv_id": f"gold{i}",
            "title": f"Gold {i}",
            "abstract": "a",
            "categories": ["cs.LG"],
            "published": "2024-01-01T00:00:00+00:00",
            "authors": ["A"],
            "url": f"http://arxiv.org/abs/gold{i}",
            "label": 1,
            # Real fixtures record which gold_query produced the paper, and write
            # all papers from one query consecutively.
            "source_query": f"q{i % queries}" if queries > 1 else "q0",
        }
        for i in range(n_gold)
    ]
    # Shipped fixtures use integer 0 for distractors; is_relevant also tolerates
    # None for hand-written pools (covered below).
    pool += [
        {
            "arxiv_id": f"dist{i}",
            "title": f"Dist {i}",
            "abstract": "b",
            "categories": ["q-bio.NC"],
            "published": "2024-01-01T00:00:00+00:00",
            "authors": ["B"],
            "url": f"http://arxiv.org/abs/dist{i}",
            "label": 0,
        }
        for i in range(n_distractor)
    ]
    return pool


class TestIsRelevant:
    def test_label_variants(self) -> None:
        assert is_relevant({"label": 1})
        assert is_relevant({"label": "1"})
        assert not is_relevant({"label": 0})  # the shipped fixture form
        assert not is_relevant({"label": None})  # defensive: hand-written pools
        assert not is_relevant({})
        assert not is_relevant({"label": "junk"})


class TestSplitRelevant:
    def test_deterministic_train_test_split(self) -> None:
        pool = _pool(n_gold=5)
        seeds, heldout = split_relevant(pool, 2)
        assert seeds == ["gold0", "gold1"]
        assert heldout == ["gold2", "gold3", "gold4"]
        assert split_relevant(pool, 2) == (seeds, heldout)  # stable across calls

    def test_seeds_are_stratified_across_source_queries(self) -> None:
        # Fixture order is query order. A file-order prefix would take every seed
        # from one query — and a single homonym query then owns the whole "liked"
        # signal (this is what made Feature 8 look unmeasurable).
        pool = _pool(n_gold=9, queries=3)
        seeds, heldout = split_relevant(pool, 3)
        strata = {p["source_query"] for p in pool if p["arxiv_id"] in set(seeds)}
        assert strata == {"q0", "q1", "q2"}  # one seed from each query
        assert set(seeds).isdisjoint(heldout)
        assert len(seeds) + len(heldout) == 9

    def test_stratified_split_is_a_partition_of_the_gold(self) -> None:
        pool = _pool(n_gold=7, queries=2)
        seeds, heldout = split_relevant(pool, 4)
        assert len(seeds) == 4
        assert sorted(seeds + heldout) == sorted(p["arxiv_id"] for p in pool if p["label"] == 1)

    def test_refuses_a_split_with_nothing_held_out(self) -> None:
        pool = _pool(n_gold=3)
        assert split_relevant(pool, 3) == ([], [])  # all gold would be seeds
        assert split_relevant(pool, 9) == ([], [])
        assert split_relevant(pool, 0) == ([], [])

    def test_ignores_distractors(self) -> None:
        seeds, heldout = split_relevant(_pool(n_gold=4, n_distractor=10), 1)
        assert all(i.startswith("gold") for i in seeds + heldout)


class TestHeldoutLabels:
    def test_seeds_are_excluded_from_evaluation(self) -> None:
        pool = _pool(n_gold=4, n_distractor=2)
        seeds, heldout = split_relevant(pool, 2)
        labels = heldout_labels(pool, seeds)

        # Seeds are training signal — scoring them would inflate every metric.
        assert not any(s in labels for s in seeds)
        assert all(labels[h] == 1 for h in heldout)
        assert labels["dist0"] == 0 and labels["dist1"] == 0
        assert sum(labels.values()) == len(heldout)

    def test_candidate_pool_drops_seeds(self) -> None:
        pool = _pool(n_gold=4, n_distractor=2)
        seeds, _ = split_relevant(pool, 2)
        ids = [p["arxiv_id"] for p in candidate_pool(pool, seeds)]
        assert not any(s in ids for s in seeds)
        assert len(ids) == len(pool) - len(seeds)


class TestBuildSeededStore:
    def test_stars_seeds_and_stores_whole_pool(self, tmp_path: Path) -> None:
        pool = _pool(n_gold=3, n_distractor=2)
        seeds, _ = split_relevant(pool, 1)
        db = tmp_path / "papers.db"
        build_seeded_store(db, pool, seeds)

        with PaperStore(db) as store:
            assert store.paper_count() == len(pool)  # FK for the embedding cache
            assert store.get_starred_papers() == seeds

    def test_seeded_store_supports_specter_cache_writes(self, tmp_path: Path) -> None:
        # paper_embeddings has a FK to papers; a starred paper missing a paper row
        # would make the SPECTER2 cache write fail (and silently kill scoring).
        import numpy as np

        from reporadar.embedding_cache import vec_to_bytes
        from reporadar.specter import SPECTER_MODEL

        pool = _pool(n_gold=2, n_distractor=1)
        seeds, _ = split_relevant(pool, 1)
        db = tmp_path / "papers.db"
        build_seeded_store(db, pool, seeds)

        with PaperStore(db) as store:
            store.save_paper_embeddings(
                [(seeds[0], SPECTER_MODEL, 3, vec_to_bytes(np.array([1, 2, 3], dtype=np.float32)))]
            )
            assert store.embedding_count(SPECTER_MODEL) == 1
