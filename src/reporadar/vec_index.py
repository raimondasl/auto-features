"""Vector KNN over cached embeddings — numpy by default, sqlite-vec when available.

Semantic search ranks papers by cosine similarity of their cached embedding to a
query vector. The numpy brute force is the always-available implementation (ample
to tens of thousands of papers); when the optional ``sqlite-vec`` extension is
installed AND this Python's sqlite3 can load extensions, ``knn`` can back the
search with a ``vec0`` cosine index for larger corpora. Both backends return the
same cosine-similarity ordering, so results are identical either way.
"""

from __future__ import annotations

import sqlite3

import numpy as np


def sqlite_vec_available() -> bool:
    """True if sqlite-vec is importable AND this sqlite3 build can load extensions.

    Many Python builds ship sqlite3 with extension loading disabled; this probes a
    throwaway in-memory connection so callers can degrade to numpy cleanly.
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.execute("SELECT vec_version()")
        finally:
            conn.close()
        return True
    except Exception:
        return False


def knn_numpy(query: np.ndarray, vectors: dict[str, np.ndarray], k: int) -> list[tuple[str, float]]:
    """Cosine-similarity top-*k* over an in-memory ``{id: vector}`` map."""
    if k <= 0 or not vectors:
        return []
    q = np.asarray(query, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return []
    ids = list(vectors)
    mat = np.vstack([np.asarray(vectors[i], dtype=np.float32) for i in ids])
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0.0] = 1.0
    sims = (mat @ q) / (norms * qn)
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def knn_sqlite_vec(
    query: np.ndarray, vectors: dict[str, np.ndarray], k: int
) -> list[tuple[str, float]]:
    """Cosine top-*k* via a temporary sqlite-vec ``vec0`` index.

    Only call when :func:`sqlite_vec_available` is True. Uses a cosine
    ``distance_metric`` (distance = ``1 - cosine``), so the returned scores are
    cosine similarities identical to :func:`knn_numpy`.
    """
    if k <= 0 or not vectors:
        return []
    import sqlite_vec

    q = np.ascontiguousarray(query, dtype=np.float32)
    dim = int(q.shape[0])
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE VIRTUAL TABLE v USING vec0"
            f"(id TEXT PRIMARY KEY, e float[{dim}] distance_metric=cosine)"
        )
        conn.executemany(
            "INSERT INTO v(id, e) VALUES (?, ?)",
            [
                (i, np.ascontiguousarray(vec, dtype=np.float32).tobytes())
                for i, vec in vectors.items()
            ],
        )
        rows = conn.execute(
            "SELECT id, distance FROM v WHERE e MATCH ? ORDER BY distance LIMIT ?",
            (q.tobytes(), k),
        ).fetchall()
        return [(str(r[0]), 1.0 - float(r[1])) for r in rows]
    finally:
        conn.close()


def knn(
    query: np.ndarray,
    vectors: dict[str, np.ndarray],
    k: int,
    prefer_sqlite_vec: bool = True,
) -> list[tuple[str, float]]:
    """Cosine top-*k* ids+scores. Uses sqlite-vec when available+preferred, else numpy.

    Any sqlite-vec hiccup falls back to numpy, which is always correct.
    """
    if prefer_sqlite_vec and sqlite_vec_available():
        try:
            return knn_sqlite_vec(query, vectors, k)
        except Exception:
            pass
    return knn_numpy(query, vectors, k)
