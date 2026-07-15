"""Tests for reporadar.vec_index — numpy KNN and (when available) sqlite-vec parity."""

from __future__ import annotations

import numpy as np

from reporadar.vec_index import knn, knn_numpy, knn_sqlite_vec, sqlite_vec_available


def _vectors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {f"id{i}": rng.standard_normal(16).astype(np.float32) for i in range(40)}


class TestKnnNumpy:
    def test_ranks_by_cosine(self) -> None:
        vectors = {
            "same": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "close": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            "orthogonal": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ids = [i for i, _ in knn_numpy(query, vectors, 3)]
        assert ids[0] == "same"
        assert ids[1] == "close"

    def test_empty_and_nonpositive_k(self) -> None:
        assert knn_numpy(np.array([1.0]), {}, 3) == []
        assert knn_numpy(np.array([1.0]), {"a": np.array([1.0])}, 0) == []

    def test_zero_norm_query(self) -> None:
        assert knn_numpy(np.array([0.0, 0.0]), {"a": np.array([1.0, 0.0])}, 1) == []

    def test_respects_k(self) -> None:
        assert len(knn_numpy(np.zeros(16) + 1, _vectors(), 5)) == 5


class TestKnnDispatch:
    def test_knn_falls_back_to_numpy(self) -> None:
        # With prefer_sqlite_vec=False, knn must equal the numpy result exactly.
        vectors, query = _vectors(), (np.zeros(16, dtype=np.float32) + 1)
        assert knn(query, vectors, 5, prefer_sqlite_vec=False) == knn_numpy(query, vectors, 5)


class TestSqliteVecParity:
    def test_same_order_and_scores_as_numpy(self) -> None:
        if not sqlite_vec_available():
            import pytest

            pytest.skip("sqlite-vec not installed / loadable in this environment")
        vectors = _vectors()
        query = np.zeros(16, dtype=np.float32) + 1
        n = knn_numpy(query, vectors, 5)
        s = knn_sqlite_vec(query, vectors, 5)
        assert [i for i, _ in n] == [i for i, _ in s]
        assert all(abs(a[1] - b[1]) < 1e-4 for a, b in zip(n, s, strict=True))
