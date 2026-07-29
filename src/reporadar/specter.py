"""SPECTER2 scientific-paper similarity — "papers near the work you liked".

all-MiniLM is a generic sentence model; **SPECTER2** is the de-facto scientific
paper embedder (citation-trained, beats SPECTER/SciNCL on SciRepEval). Semantic
Scholar serves precomputed SPECTER2 vectors for free, so RepoRadar gets
domain-tuned similarity with **no local model** — nothing to download, no
``adapters``/transformers pin.

The query side is the neat part: instead of embedding the repo profile (which
would require running SPECTER2 locally), the query is the **centroid of the
papers you starred or rated highly**, in the same citation-trained space. So the
component answers "how close is this paper to the work I already valued?".

Two properties this module exists to get right:

- **Vectors are cached** in the store's ``paper_embeddings`` table under a distinct
  ``specter_v2`` model key, so 768-dim SPECTER2 vectors can never mix with the
  384-dim MiniLM vectors used by :mod:`reporadar.semantic` (different keys,
  different rows) and each paper is fetched once.
- **Scores are pool-normalized.** Raw SPECTER2 cosines sit in a narrow band
  (~0.87–0.93 even between unrelated papers), so the raw value carries almost no
  ranking signal. Min-max normalizing across the run's candidates restores the
  spread — the same trick :func:`reporadar.citations.normalize_citations` uses.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

from reporadar.citations import _s2_batch_post, _s2_id
from reporadar.embedding_cache import vec_from_bytes, vec_to_bytes
from reporadar.store import PaperStore

logger = logging.getLogger(__name__)

# Cache key for SPECTER2 vectors; distinct from the MiniLM key so the two
# dimensionalities are stored (and read back) separately.
SPECTER_MODEL = "specter_v2"
# Papers S2 has no vector for are recorded under this key so they aren't
# re-requested on every run (cleared by `rr update --rebuild-embeddings`).
SPECTER_MISS_MODEL = "specter_v2:miss"
SPECTER_DIM = 768
# Much smaller than the citations batch: each entry carries 768 floats, so a
# 500-id chunk would be a multi-MB response under a timeout tuned for tiny ones.
_S2_BATCH_LIMIT = 100
# Below this many scoreable papers, or this much raw-cosine spread, min-max
# normalization would amplify noise into the component's full dynamic range —
# so the component is dropped instead (measured: unrelated papers still sit
# ~0.87–0.93, and float32 cosine noise is ~1e-7).
_MIN_POOL = 3
_MIN_SPREAD = 0.01
# Reading the whole model cache beats binding this many parameters.
_MAX_ID_PARAMS = 900


def fetch_specter_vectors(
    arxiv_ids: list[str], api_key: str | None = None
) -> dict[str, np.ndarray]:
    """Fetch precomputed SPECTER2 vectors from Semantic Scholar.

    Returns ``{arxiv_id: vector}`` for the papers S2 knows; ids it can't resolve
    are simply absent. Returns ``{}`` on API failure (graceful degradation).
    """
    ids = [a for a in dict.fromkeys(arxiv_ids) if ":" not in a]  # real arXiv ids only
    if not ids:
        return {}

    out: dict[str, np.ndarray] = {}
    for start in range(0, len(ids), _S2_BATCH_LIMIT):
        chunk = ids[start : start + _S2_BATCH_LIMIT]
        data = _s2_batch_post(
            [_s2_id(a) for a in chunk],
            "embedding.specter_v2",
            api_key,
            max_retries=3,
            base_delay=2.0,
        )
        if data is None:
            continue
        for i, entry in enumerate(data):
            embedding = (entry or {}).get("embedding") or {}
            vector = embedding.get("vector")
            if vector:
                out[chunk[i]] = np.asarray(vector, dtype=np.float32)
    return out


def _cached_blobs(store: PaperStore, model: str, ids: list[str]) -> dict[str, bytes]:
    """Cached blobs for *ids*, avoiding a whole-cache read when the list is small."""
    if len(ids) > _MAX_ID_PARAMS:
        return store.get_embedding_blobs(model)
    return store.get_embedding_blobs(model, ids)


def _persist(store: PaperStore, rows: list[tuple[str, str, int, bytes]]) -> None:
    """Write cache rows, skipping any paper that isn't in the corpus.

    ``paper_embeddings`` has a foreign key to ``papers``, while stars/ratings do
    not — so a seed id with no paper row would raise IntegrityError and, since the
    caller only guards the whole stage, would silently disable SPECTER2 scoring on
    every future run. A cache write must never be able to break scoring.
    """
    safe = [r for r in rows if store.get_paper(r[0]) is not None]
    if not safe:
        return
    try:
        store.save_paper_embeddings(safe)
    except sqlite3.IntegrityError as exc:  # pragma: no cover - defensive
        logger.warning("Could not cache SPECTER2 vectors: %s", exc)


def load_or_fetch_vectors(
    store: PaperStore, arxiv_ids: list[str], api_key: str | None = None
) -> dict[str, np.ndarray]:
    """Return cached SPECTER2 vectors, fetching (and caching) any that are missing.

    Papers S2 has no vector for are recorded as misses so they aren't re-requested
    every run; `rr update --rebuild-embeddings` clears both caches to retry them.
    """
    wanted = list(dict.fromkeys(arxiv_ids))
    if not wanted:
        return {}
    cached_blobs = _cached_blobs(store, SPECTER_MODEL, wanted)
    known_misses = set(_cached_blobs(store, SPECTER_MISS_MODEL, wanted))

    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for arxiv_id in wanted:
        blob = cached_blobs.get(arxiv_id)
        if blob:
            out[arxiv_id] = vec_from_bytes(blob)
        elif arxiv_id not in known_misses:
            missing.append(arxiv_id)

    if missing:
        fetched = fetch_specter_vectors(missing, api_key=api_key)
        rows = [
            (arxiv_id, SPECTER_MODEL, int(vec.shape[0]), vec_to_bytes(vec))
            for arxiv_id, vec in fetched.items()
        ]
        # Remember the ids S2 had nothing for, so the next run doesn't re-ask.
        rows += [(a, SPECTER_MISS_MODEL, 0, b"") for a in missing if a not in fetched]
        _persist(store, rows)
        out.update(fetched)
    return out


def build_query_vector(
    store: PaperStore, api_key: str | None = None, min_rating: int = 4
) -> np.ndarray | None:
    """Centroid of the SPECTER2 vectors of papers the user starred / rated highly.

    Returns None when there are no usable seeds (so the caller can skip the whole
    stage) — this component is a "more like what I liked" signal and is meaningless
    without at least one liked paper.
    """
    seeds: list[str] = list(store.get_starred_papers())
    seeds += [aid for aid, rating in store.get_all_ratings().items() if rating >= min_rating]
    seeds = [s for s in dict.fromkeys(seeds) if ":" not in s]
    if not seeds:
        return None

    vectors = load_or_fetch_vectors(store, seeds, api_key=api_key)
    # L2-normalize before averaging: only the centroid's *direction* matters
    # downstream (everything is cosine), and averaging raw vectors would let a
    # seed with a larger norm dominate the mean direction.
    usable = [
        v / n
        for v in vectors.values()
        if v.size == SPECTER_DIM and (n := float(np.linalg.norm(v))) > 0.0
    ]
    if not usable:
        return None
    centroid: np.ndarray = np.mean(np.vstack(usable), axis=0)
    return centroid


def specter_scores(
    query: np.ndarray,
    vectors: dict[str, np.ndarray],
    min_pool: int = _MIN_POOL,
    min_spread: float = _MIN_SPREAD,
) -> dict[str, float]:
    """Pool-normalized SPECTER2 similarity in ``[0, 1]`` per paper.

    Cosine similarity to *query*, min-max normalized across the pool. The
    normalization is what makes the component usable at all — raw SPECTER2 cosines
    cluster in a narrow band (~0.87–0.93 even between unrelated papers), so
    unnormalized they carry almost no ranking signal.

    But min-max also *manufactures* a full-range signal from nothing, so it is only
    applied when the pool can actually support it: fewer than *min_pool* scoreable
    papers, or a raw spread below *min_spread*, returns ``{}`` — the component is
    dropped rather than amplifying noise (or handing a lone paper a free 1.0).
    """
    qn = float(np.linalg.norm(query))
    if qn == 0.0 or not vectors:
        return {}

    sims: dict[str, float] = {}
    for arxiv_id, vec in vectors.items():
        if vec.size != query.size:
            continue  # never mix dimensionalities (e.g. a stale MiniLM vector)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            continue
        sims[arxiv_id] = float(np.dot(vec, query) / (norm * qn))

    if len(sims) < min_pool:
        return {}
    lo, hi = min(sims.values()), max(sims.values())
    if hi - lo < min_spread:
        return {}
    return {k: (v - lo) / (hi - lo) for k, v in sims.items()}


def score_papers(
    store: PaperStore,
    papers: list[dict[str, Any]],
    api_key: str | None = None,
) -> dict[str, float]:
    """End-to-end: seeds -> query centroid -> cached/fetched vectors -> scores.

    Returns ``{}`` when the user has no liked papers, the pool carries no usable
    signal, or S2 is unreachable — the caller then ranks without this component.

    Papers S2 has no vector for (non-arXiv sources, or simply not indexed yet) are
    filled with the pool **mean** rather than left out. Omitting them would drop
    the component from their weighted average, which — because min-max always puts
    some real paper at 0.0 — would systematically favour papers S2 knows nothing
    about over papers it does. A neutral fill keeps every paper on the same scale.
    """
    query = build_query_vector(store, api_key=api_key)
    if query is None:
        return {}
    ids = [p["arxiv_id"] for p in papers]
    vectors = load_or_fetch_vectors(store, ids, api_key=api_key)
    scores = specter_scores(query, vectors)
    if not scores:
        return {}
    neutral = sum(scores.values()) / len(scores)
    return {p["arxiv_id"]: scores.get(p["arxiv_id"], neutral) for p in papers}
