"""Persistent per-paper embedding cache — compute a vector once, reuse it forever.

The ranker's embedding-similarity path re-encoded *every* candidate paper on *every*
run. This caches each paper's vector in the store (``paper_embeddings``, keyed by
``(arxiv_id, model)``), so only papers never embedded before are encoded — the
deferred ``sqlite-vec`` half of Feature 4, realized as a plain-BLOB cache that
needs no native extension to read.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reporadar import embeddings
from reporadar.store import PaperStore

# Vectors are stored as raw little-endian float32 bytes; dim is implied by length.
_DTYPE = np.float32


# Bump the recipe suffix if the encoded text (embeddings.compute_paper_embedding)
# or the model changes, so stale cached vectors are ignored instead of served.
# (Or clear at runtime with `rr update --rebuild-embeddings`.)
_RECIPE = "v1"


def default_model() -> str:
    """The cache key for the current embedding model + recipe (see _RECIPE)."""
    return f"{embeddings._MODEL_NAME}:{_RECIPE}"


def vec_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a vector to the cache's byte layout (contiguous float32)."""
    return np.ascontiguousarray(vec, dtype=_DTYPE).tobytes()


def vec_from_bytes(blob: bytes) -> np.ndarray:
    """Deserialize a cached blob back to a 1-D float32 vector."""
    return np.frombuffer(blob, dtype=_DTYPE)


def _safe_vec(blob: bytes | None) -> np.ndarray | None:
    """Deserialize a blob, returning None for a missing/corrupt (non-float32) one."""
    if not blob or len(blob) % 4 != 0:
        return None
    return np.frombuffer(blob, dtype=_DTYPE)


def load_cached_vectors(store: PaperStore, model: str | None = None) -> dict[str, np.ndarray]:
    """Return every cached ``{arxiv_id: vector}`` for *model* (no computation)."""
    model = model or default_model()
    out: dict[str, np.ndarray] = {}
    for aid, blob in store.get_embedding_blobs(model).items():
        vec = _safe_vec(blob)
        if vec is not None:
            out[aid] = vec
    return out


def embed_papers_cached(
    store: PaperStore,
    papers: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, np.ndarray]:
    """Return ``{arxiv_id: vector}`` for *papers*, encoding only the uncached ones.

    Cached vectors are read from the store (a corrupt/wrong-length blob is treated
    as uncached and re-encoded); any paper without a usable cached vector is encoded
    once via :mod:`reporadar.embeddings` (which requires the ``embeddings`` extra).
    Newly encoded vectors are persisted in a single batched commit.
    """
    model = model or default_model()
    cached = store.get_embedding_blobs(model)  # whole cache for the model
    out: dict[str, np.ndarray] = {}
    new_rows: list[tuple[str, str, int, bytes]] = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        vec = _safe_vec(cached.get(arxiv_id))
        if vec is not None:
            out[arxiv_id] = vec
            continue
        vec = np.ascontiguousarray(embeddings.compute_paper_embedding(paper), dtype=_DTYPE)
        out[arxiv_id] = vec
        new_rows.append((arxiv_id, model, int(vec.shape[0]), vec.tobytes()))
    store.save_paper_embeddings(new_rows)
    return out
