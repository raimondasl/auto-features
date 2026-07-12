"""Hybrid retrieval: fuse the heuristic ranking with a lexical BM25 ranking.

The heuristic ranker scores papers by weighted keyword/category/recency overlap.
That misses papers where the repo's vocabulary and the paper's vocabulary differ
(a real, actionable paper phrased in unfamiliar terms ranks low). BM25 is a
strong lexical baseline that weights rare, discriminative terms; fusing the two
rank orders with Reciprocal Rank Fusion (RRF) — which combines *ranks*, not
scores, so the two signals' incompatible scales don't matter — surfaces papers
either signal alone would bury.

The corpus is per-run (the fetched candidates, tens of papers), so a plain-Python
BM25 is ample and keeps this dependency-free. RRF preserves each paper's original
``score_total`` (for downstream thresholds) and drives ordering by ``rrf_score``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from reporadar.profiler import RepoProfile

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _profile_query(profile: RepoProfile) -> list[str]:
    """Bag-of-words query from the repo profile (keywords + anchors + domains)."""
    terms: list[str] = []
    for term, _ in list(getattr(profile, "keywords", []))[:20]:
        terms += _tokenize(str(term))
    for anchor in list(getattr(profile, "anchors", []))[:20]:
        terms += _tokenize(str(anchor))
    for domain in list(getattr(profile, "domains", []))[:10]:
        terms += _tokenize(str(domain))
    return terms


def bm25_scores(
    query: list[str],
    corpus: list[list[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Okapi BM25 score of each tokenized document against the query bag."""
    n = len(corpus)
    if n == 0:
        return []
    doc_len = [len(d) for d in corpus]
    avgdl = sum(doc_len) / n if n else 0.0

    tf: list[Counter[str]] = [Counter(d) for d in corpus]
    df: dict[str, int] = {}
    for counts in tf:
        for term in counts:
            df[term] = df.get(term, 0) + 1

    q_terms = set(query)
    # BM25+ idf: always positive, so a term in every doc still contributes ~0, not negative.
    idf = {t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in q_terms}

    scores: list[float] = []
    for i, counts in enumerate(tf):
        norm = k1 * (1 - b + b * (doc_len[i] / avgdl if avgdl else 0.0))
        s = 0.0
        for t in q_terms:
            f = counts.get(t, 0)
            if f:
                s += idf[t] * (f * (k1 + 1)) / (f + norm)
        scores.append(s)
    return scores


def bm25_ranked_ids(papers: list[dict[str, Any]], profile: RepoProfile) -> list[str]:
    """arxiv_ids ordered by BM25 relevance of title+abstract to the repo profile."""
    query = _profile_query(profile)
    corpus = [_tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in papers]
    scores = bm25_scores(query, corpus)
    order = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)
    return [str(papers[i]["arxiv_id"]) for i in order]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: each id scores sum of 1/(k + rank) across rankings.

    Rank is 0-based, so the top of each list contributes 1/(k+1). Ids absent from
    a ranking simply don't accrue from it. ``k=60`` is the standard default.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank + 1)
    return fused


def hybrid_reorder(
    scores: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    profile: RepoProfile,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Re-order heuristic score dicts by RRF fusion with a BM25 lexical ranking.

    *scores* are the heuristic ranker's output (assumed sorted by ``score_total``
    descending); *papers* are the corresponding paper dicts. Each score dict gets
    an ``rrf_score`` and the list is re-sorted by it. ``score_total`` is left
    intact so downstream threshold logic is unaffected — only the order changes.
    """
    heuristic_order = [str(s["arxiv_id"]) for s in scores]
    bm25_order = bm25_ranked_ids(papers, profile)
    fused = rrf_fuse([heuristic_order, bm25_order], k=k)
    for s in scores:
        s["rrf_score"] = round(fused.get(str(s["arxiv_id"]), 0.0), 6)
    return sorted(scores, key=lambda s: s["rrf_score"], reverse=True)
