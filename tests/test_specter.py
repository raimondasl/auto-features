"""Tests for reporadar.specter — SPECTER2 similarity to the work you liked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reporadar.specter import (
    SPECTER_DIM,
    SPECTER_MODEL,
    build_query_vector,
    fetch_specter_vectors,
    load_or_fetch_vectors,
    score_papers,
    specter_scores,
)
from reporadar.store import PaperStore


def _paper(arxiv_id: str, title: str = "T") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["A"],
        "abstract": "abstract",
        "categories": ["cs.LG"],
        "published": "2024-01-01T00:00:00+00:00",
        "updated": None,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": None,
    }


def _vec(*values: float) -> list[float]:
    """A SPECTER_DIM-long vector whose leading entries are *values*."""
    out = [0.0] * SPECTER_DIM
    for i, v in enumerate(values):
        out[i] = v
    return out


class TestFetchSpecterVectors:
    @patch("reporadar.specter._s2_batch_post")
    def test_parses_vectors_and_skips_missing(self, mock_post: MagicMock) -> None:
        mock_post.return_value = [
            {"embedding": {"model": "specter_v2", "vector": _vec(1.0, 0.0)}},
            {"embedding": None},  # S2 has the paper but no vector
            None,  # unknown paper
        ]
        out = fetch_specter_vectors(["2401.1v1", "2401.2v1", "2401.3v1"])
        assert list(out) == ["2401.1v1"]
        assert out["2401.1v1"].shape == (SPECTER_DIM,)
        assert mock_post.call_args[0][1] == "embedding.specter_v2"

    @patch("reporadar.specter._s2_batch_post")
    def test_drops_synthetic_ids(self, mock_post: MagicMock) -> None:
        # Only real arXiv ids resolve at S2 — don't waste batch slots.
        assert fetch_specter_vectors(["dblp:x/y", "biorxiv:10.1/z", "ss:abc"]) == {}
        mock_post.assert_not_called()

    @patch("reporadar.specter._s2_batch_post")
    def test_api_failure_returns_empty(self, mock_post: MagicMock) -> None:
        mock_post.return_value = None
        assert fetch_specter_vectors(["2401.1v1"]) == {}


class TestLoadOrFetch:
    @patch("reporadar.specter.fetch_specter_vectors")
    def test_caches_then_reuses(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2401.1v1"))
            mock_fetch.return_value = {"2401.1v1": np.array(_vec(1.0), dtype=np.float32)}

            first = load_or_fetch_vectors(store, ["2401.1v1"])
            assert mock_fetch.call_count == 1
            assert store.embedding_count(SPECTER_MODEL) == 1

            mock_fetch.reset_mock()
            second = load_or_fetch_vectors(store, ["2401.1v1"])
            mock_fetch.assert_not_called()  # served from cache
            assert np.allclose(first["2401.1v1"], second["2401.1v1"])

    @patch("reporadar.specter.fetch_specter_vectors")
    def test_fetches_only_the_missing_ones(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2401.1v1"))
            store.upsert_paper(_paper("2401.2v1"))
            mock_fetch.return_value = {"2401.1v1": np.array(_vec(1.0), dtype=np.float32)}
            load_or_fetch_vectors(store, ["2401.1v1"])

            mock_fetch.reset_mock()
            mock_fetch.return_value = {"2401.2v1": np.array(_vec(0.0, 1.0), dtype=np.float32)}
            out = load_or_fetch_vectors(store, ["2401.1v1", "2401.2v1"])
            assert mock_fetch.call_args[0][0] == ["2401.2v1"]  # only the uncached one
            assert set(out) == {"2401.1v1", "2401.2v1"}


class TestBuildQueryVector:
    @patch("reporadar.specter.fetch_specter_vectors")
    def test_centroid_of_starred_and_highly_rated(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            for aid in ("2401.1v1", "2401.2v1", "2401.3v1"):
                store.upsert_paper(_paper(aid))
            store.star_paper("2401.1v1")
            store.save_rating("2401.2v1", 5)  # seed
            store.save_rating("2401.3v1", 2)  # not a seed
            mock_fetch.return_value = {
                "2401.1v1": np.array(_vec(2.0), dtype=np.float32),
                "2401.2v1": np.array(_vec(0.0, 2.0), dtype=np.float32),
            }
            query = build_query_vector(store)

        assert query is not None
        assert mock_fetch.call_args[0][0] == ["2401.1v1", "2401.2v1"]  # low rating excluded
        # Seeds are L2-normalized before averaging (only the centroid's direction
        # matters downstream), so two orthogonal unit seeds average to 0.5 each —
        # a raw mean would have been magnitude-weighted (1.0 here).
        assert query[0] == pytest.approx(0.5) and query[1] == pytest.approx(0.5)

    @patch("reporadar.specter.fetch_specter_vectors")
    def test_large_norm_seed_does_not_dominate(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            for aid in ("2401.1v1", "2401.2v1"):
                store.upsert_paper(_paper(aid))
                store.star_paper(aid)
            mock_fetch.return_value = {
                "2401.1v1": np.array(_vec(100.0), dtype=np.float32),  # huge norm
                "2401.2v1": np.array(_vec(0.0, 1.0), dtype=np.float32),
            }
            query = build_query_vector(store)

        assert query is not None
        assert query[0] == pytest.approx(query[1])  # equal influence despite 100x norm

    def test_none_without_seeds(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            assert build_query_vector(store) is None

    @patch("reporadar.specter.fetch_specter_vectors")
    def test_none_when_s2_has_no_vectors(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2401.1v1"))
            store.star_paper("2401.1v1")
            mock_fetch.return_value = {}
            assert build_query_vector(store) is None


class TestSpecterScores:
    def test_pool_normalized_to_full_range(self) -> None:
        # Raw SPECTER2 cosines cluster near 0.9; normalization is what makes the
        # component discriminate at all.
        query = np.array(_vec(1.0), dtype=np.float32)
        vectors = {
            "near": np.array(_vec(1.0, 0.05), dtype=np.float32),
            "mid": np.array(_vec(1.0, 0.5), dtype=np.float32),
            "far": np.array(_vec(1.0, 1.0), dtype=np.float32),
        }
        scores = specter_scores(query, vectors)
        assert scores["near"] == 1.0 and scores["far"] == 0.0
        assert 0.0 < scores["mid"] < 1.0

    def test_drops_component_for_tiny_pool(self) -> None:
        # Min-max on one or two papers manufactures a full-range signal from
        # nothing — a lone paper would get a free 1.0.
        query = np.array(_vec(1.0), dtype=np.float32)
        near = np.array(_vec(1.0, 0.05), dtype=np.float32)
        far = np.array(_vec(1.0, 1.0), dtype=np.float32)
        assert specter_scores(query, {"only": near}) == {}
        assert specter_scores(query, {"a": near, "b": far}) == {}

    def test_drops_component_when_spread_is_noise(self) -> None:
        # A pool whose cosines differ only by float noise carries no signal.
        query = np.array(_vec(1.0), dtype=np.float32)
        vectors = {f"p{i}": np.array(_vec(1.0, 1e-7 * i), dtype=np.float32) for i in range(5)}
        assert specter_scores(query, vectors) == {}

    def test_ties_drop_the_component(self) -> None:
        query = np.array(_vec(1.0), dtype=np.float32)
        v = np.array(_vec(1.0), dtype=np.float32)
        assert specter_scores(query, {"a": v, "b": v, "c": v}) == {}

    def test_skips_mismatched_dimensions(self) -> None:
        # A 384-dim MiniLM vector must never be compared against a 768-dim query.
        query = np.array(_vec(1.0), dtype=np.float32)
        vectors = {
            "ok1": np.array(_vec(1.0, 0.05), dtype=np.float32),
            "ok2": np.array(_vec(1.0, 0.5), dtype=np.float32),
            "ok3": np.array(_vec(1.0, 1.0), dtype=np.float32),
            "wrong_dim": np.ones(384, dtype=np.float32),
        }
        assert set(specter_scores(query, vectors)) == {"ok1", "ok2", "ok3"}

    def test_empty_inputs(self) -> None:
        assert specter_scores(np.zeros(SPECTER_DIM, dtype=np.float32), {}) == {}
        assert specter_scores(np.array(_vec(1.0), dtype=np.float32), {}) == {}


class TestScorePapers:
    def test_no_seeds_returns_empty(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            assert score_papers(store, [_paper("2401.9v1")]) == {}

    @patch("reporadar.specter.fetch_specter_vectors")
    def test_end_to_end(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        spread = {
            "2401.near": _vec(1.0, 0.0),
            "2401.mid": _vec(1.0, 0.6),
            "2401.far": _vec(0.0, 1.0),
        }
        with PaperStore(tmp_path / "p.db") as store:
            for aid in ("2401.seed", *spread):
                store.upsert_paper(_paper(aid))
            store.star_paper("2401.seed")
            mock_fetch.side_effect = lambda ids, api_key=None: {
                aid: np.array(spread.get(aid, _vec(1.0, 0.0)), dtype=np.float32) for aid in ids
            }
            scores = score_papers(store, [_paper(a) for a in spread])

        assert scores["2401.near"] == 1.0
        assert scores["2401.far"] == 0.0
        assert 0.0 < scores["2401.mid"] < 1.0

    @patch("reporadar.specter.fetch_specter_vectors")
    def test_papers_without_vectors_get_the_neutral_mean(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        # Otherwise min-max (which always puts some real paper at 0.0) would make
        # papers S2 knows nothing about outrank papers it does.
        known = {"2401.a": _vec(1.0, 0.0), "2401.b": _vec(1.0, 0.6), "2401.c": _vec(0.0, 1.0)}
        with PaperStore(tmp_path / "p.db") as store:
            for aid in ("2401.seed", *known):
                store.upsert_paper(_paper(aid))
            store.star_paper("2401.seed")
            mock_fetch.side_effect = lambda ids, api_key=None: {
                aid: np.array(known.get(aid, _vec(1.0, 0.0)), dtype=np.float32)
                for aid in ids
                if aid in known or aid == "2401.seed"
            }
            papers = [_paper(a) for a in known] + [_paper("dblp:conf/x/y")]
            scores = score_papers(store, papers)

        assert set(scores) == {*known, "dblp:conf/x/y"}
        known_scores = [scores[a] for a in known]
        assert scores["dblp:conf/x/y"] == sum(known_scores) / len(known_scores)


class TestCacheResilience:
    @patch("reporadar.specter.fetch_specter_vectors")
    def test_orphan_seed_does_not_break_scoring(
        self, mock_fetch: MagicMock, tmp_path: Path
    ) -> None:
        # A rating/star can name a paper that isn't in `papers` (paper_stars has no
        # FK). Caching its vector would raise IntegrityError and — because the CLI
        # only guards the whole stage — silently kill SPECTER2 on every later run.
        with PaperStore(tmp_path / "p.db") as store:
            store.save_rating("2401.orphan", 5)  # no papers row
            mock_fetch.return_value = {"2401.orphan": np.array(_vec(1.0), dtype=np.float32)}
            out = load_or_fetch_vectors(store, ["2401.orphan"])

        assert "2401.orphan" in out  # still usable in-memory
        assert build_query_vector.__name__  # sanity: module import intact

    @patch("reporadar.specter.fetch_specter_vectors")
    def test_misses_are_not_refetched(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2401.unknown"))
            mock_fetch.return_value = {}  # S2 has no vector for it
            load_or_fetch_vectors(store, ["2401.unknown"])
            assert mock_fetch.call_count == 1

            mock_fetch.reset_mock()
            load_or_fetch_vectors(store, ["2401.unknown"])
            mock_fetch.assert_not_called()  # negative-cached
