"""Semantic (embedding) search over the cached corpus, with optional BM25 fusion.

Complements the lexical :mod:`reporadar.search`: embeds the query and ranks papers
by cosine similarity of their cached vectors (:mod:`reporadar.embedding_cache` +
:mod:`reporadar.vec_index`). ``hybrid`` fuses the semantic and BM25 orders with
Reciprocal Rank Fusion — the same fusion the ranker's hybrid mode uses. Requires
the ``embeddings`` extra (to encode the query and any uncached papers).
"""

from __future__ import annotations

from typing import Any

from reporadar import embeddings, vec_index
from reporadar.embedding_cache import default_model, embed_papers_cached
from reporadar.retrieval import rrf_fuse
from reporadar.search import search_corpus
from reporadar.store import PaperStore

# When fusing, pull a deeper candidate pool from each ranker before combining.
_FUSION_POOL = 50


def semantic_search(
    store: PaperStore,
    query: str,
    limit: int = 20,
    hybrid: bool = False,
    prefer_sqlite_vec: bool = True,
    papers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Rank papers by embedding similarity to *query* (optionally fused with BM25).

    Encodes any uncached papers once (persisting them). Returns paper dicts with a
    ``search_score`` (cosine similarity, or the fused RRF score when *hybrid*),
    best-first, up to *limit*. Raises RuntimeError if the ``embeddings`` extra is
    missing (needed to encode the query).
    """
    if limit <= 0 or not query.strip():
        return []
    if papers is None:
        papers = store.get_all_papers()
    if not papers:
        return []

    vectors = embed_papers_cached(store, papers, default_model())
    query_vec = embeddings.compute_embedding(query)
    by_id = {p["arxiv_id"]: p for p in papers}

    pool = max(limit, _FUSION_POOL) if hybrid else limit
    semantic = vec_index.knn(query_vec, vectors, pool, prefer_sqlite_vec=prefer_sqlite_vec)

    if not hybrid:
        out = [
            {**by_id[aid], "search_score": round(float(score), 4)}
            for aid, score in semantic
            if aid in by_id
        ]
        return out[:limit]

    # Hybrid: fuse the semantic order with the BM25 lexical order via RRF (rank-
    # based, so the incompatible cosine/BM25 score scales don't matter).
    semantic_ids = [aid for aid, _ in semantic]
    lexical_ids = [p["arxiv_id"] for p in search_corpus(papers, query, limit=pool)]
    fused = rrf_fuse([semantic_ids, lexical_ids])
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [
        {**by_id[aid], "search_score": round(float(score), 4)}
        for aid, score in ranked
        if aid in by_id
    ]
