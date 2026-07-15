"""Full-corpus free-text search over every stored paper (dependency-free BM25).

``rr update`` fetches a fresh candidate set each run; over time the SQLite store
becomes a personal corpus of everything ever fetched. ``rr search "<query>"``
runs Okapi BM25 (reusing :mod:`reporadar.retrieval`) over that whole corpus — a
local, offline, queryable index with no embeddings or network required.

This rebuilds the BM25 index in memory on every call (cost is O(corpus tokens)),
which is intended for corpora up to the low thousands of papers. A persistent
lexical/vector index (the deferred ``sqlite-vec`` half of Feature 4) is future work.
"""

from __future__ import annotations

from typing import Any

from reporadar.retrieval import _tokenize, bm25_scores


def search_corpus(
    papers: list[dict[str, Any]],
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return up to *limit* papers most relevant to *query*, best-first.

    Each returned dict is a copy of the paper with an added ``search_score`` (BM25
    over title + abstract). Papers with no query-term overlap (score 0) are
    dropped; a blank query or empty corpus returns ``[]``.
    """
    q = _tokenize(query)
    if not q or not papers or limit <= 0:
        return []

    corpus = [_tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in papers]
    scores = bm25_scores(q, corpus)

    ranked = sorted(
        ((paper, score) for paper, score in zip(papers, scores, strict=True) if score > 0),
        key=lambda ps: ps[1],
        reverse=True,
    )
    return [{**paper, "search_score": round(score, 4)} for paper, score in ranked[:limit]]
