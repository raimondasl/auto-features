"""Seeded-preference evaluation — makes the personalization features falsifiable.

Three shipped ranking signals key off the user's *own* starred / highly-rated
papers: SPECTER2 similarity (Feature 7), citation proximity (Feature 8) and
learned recommendations (Feature 5). The Tier B harness ranks a candidate pool
directly and never builds a store, so it has no stars or ratings — those
components never fire and were shipped **tested but not benchmarked**.

This module supplies the missing ingredient with a clean train/test split over
the Tier A *labeled* fixtures:

1. Take a case's known-relevant papers (``label == 1``).
2. **Seed** a few of them as "papers the user starred" — the training signal.
3. **Hold out** the rest and score how well each component ranks *those*.

Seeds are removed from the candidate pool, so a component can't win by matching
the very papers it was given. Unlike Tier B this is **deterministic** — no LLM
judge, so there is no noise floor to fight; SPECTER2 vectors and reference lists
are stable per paper and cached in the store after the first fetch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reporadar.store import PaperStore

RELEVANT_LABEL = 1


def is_relevant(paper: dict[str, Any]) -> bool:
    """True when a fixture paper is labeled relevant (gold)."""
    try:
        return int(paper.get("label") or 0) == RELEVANT_LABEL
    except (TypeError, ValueError):
        return False


def split_relevant(pool: list[dict[str, Any]], n_seeds: int) -> tuple[list[str], list[str]]:
    """Split the pool's relevant papers into ``(seed_ids, heldout_ids)``.

    Seeds are drawn **round-robin across the distinct ``source_query`` values** the
    fixture was built from, not simply the first *n* in file order. Fixture order is
    gold-query order, so taking a prefix draws every seed from one query — and if
    that query is a keyword homonym (``rag``'s first query yields cs.HC papers about
    "augmented" cognition) the entire "papers the user liked" signal is off-domain.
    Stratifying makes the seed set span the case's sub-topics.

    Deterministic (queries and papers keep fixture order within each stratum).
    Returns empty lists when there aren't enough relevant papers to leave at least
    one held out — a split with nothing to measure is not a measurement.
    """
    relevant = [p for p in pool if is_relevant(p)]
    if n_seeds < 1 or len(relevant) <= n_seeds:
        return [], []

    by_query: dict[str, list[str]] = {}
    for paper in relevant:
        by_query.setdefault(str(paper.get("source_query") or ""), []).append(paper["arxiv_id"])

    seeds: list[str] = []
    round_index = 0
    while len(seeds) < n_seeds:
        added = False
        for ids in by_query.values():
            if round_index < len(ids):
                seeds.append(ids[round_index])
                added = True
                if len(seeds) == n_seeds:
                    break
        if not added:  # pragma: no cover - guarded by the length check above
            break
        round_index += 1

    chosen = set(seeds)
    heldout = [p["arxiv_id"] for p in relevant if p["arxiv_id"] not in chosen]
    return seeds, heldout


def heldout_labels(pool: list[dict[str, Any]], seed_ids: list[str]) -> dict[str, int]:
    """Labels for evaluation: held-out relevant papers are 1, distractors 0.

    Seed papers are omitted entirely — they are the training signal, not test
    items, and leaving them in would inflate every metric.
    """
    seeds = set(seed_ids)
    return {
        p["arxiv_id"]: (RELEVANT_LABEL if is_relevant(p) else 0)
        for p in pool
        if p["arxiv_id"] not in seeds
    }


def candidate_pool(pool: list[dict[str, Any]], seed_ids: list[str]) -> list[dict[str, Any]]:
    """The papers to rank: everything except the seeds."""
    seeds = set(seed_ids)
    return [p for p in pool if p["arxiv_id"] not in seeds]


def build_seeded_store(
    db_path: str | Path, pool: list[dict[str, Any]], seed_ids: list[str]
) -> None:
    """Create a store holding *pool* with *seed_ids* starred.

    The whole pool is upserted (not just the seeds) because ``paper_embeddings``
    has a foreign key to ``papers`` — a starred paper with no paper row would make
    the SPECTER2 cache write fail.
    """
    with PaperStore(db_path) as store:
        store.upsert_papers([_storable(p) for p in pool])
        for arxiv_id in seed_ids:
            store.star_paper(arxiv_id)


def _storable(paper: dict[str, Any]) -> dict[str, Any]:
    """Fixture paper -> the store's expected paper dict."""
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "categories": paper.get("categories", []),
        "published": paper.get("published") or "2024-01-01T00:00:00+00:00",
        "updated": paper.get("updated"),
        "url": paper.get("url") or f"http://arxiv.org/abs/{paper['arxiv_id']}",
        "pdf_url": paper.get("pdf_url"),
    }


def specter_component(
    db_path: str | Path, candidates: list[dict[str, Any]], api_key: str | None = None
) -> tuple[dict[str, float], dict[str, Any]]:
    """SPECTER2 scores for *candidates*, seeded from the store's stars.

    Returns ``(scores, diagnostics)``. The diagnostics matter: ``score_papers``
    neutral-fills candidates S2 has no vector for, so ``len(scores)`` is always the
    candidate count and cannot reveal poor coverage — and an S2 outage returns the
    same empty dict as a genuine absence of signal.
    """
    from reporadar.specter import build_query_vector, load_or_fetch_vectors, score_papers

    with PaperStore(db_path) as store:
        query = build_query_vector(store, api_key=api_key)
        vectors = load_or_fetch_vectors(store, [p["arxiv_id"] for p in candidates], api_key=api_key)
        scores = score_papers(store, candidates, api_key=api_key)
    return scores, {
        "has_query_vector": query is not None,
        "n_vectors": len(vectors),
        "n_candidates": len(candidates),
        "coverage": round(len(vectors) / len(candidates), 4) if candidates else 0.0,
    }


def citation_proximity_component(
    db_path: str | Path, candidates: list[dict[str, Any]], api_key: str | None = None
) -> tuple[dict[str, float], dict[str, Any]]:
    """Citation-proximity scores: 1.0 for candidates citing a starred paper.

    Mirrors what ``rr update`` does (build seed set -> fetch references -> match)
    and persists the edges. Returns ``(scores, diagnostics)`` so "no candidate cites
    a seed" is distinguishable from "Semantic Scholar was unreachable" — the whole
    Feature 8 conclusion rests on telling those apart.
    """
    from reporadar.citation_graph import build_seed_set, find_citation_links
    from reporadar.citations import fetch_references

    with PaperStore(db_path) as store:
        seeds = build_seed_set(store)
        if not seeds:
            return {}, {"n_seeds": 0, "papers_with_refs": 0, "ref_edges": 0}
        ids = [p["arxiv_id"] for p in candidates if ":" not in p["arxiv_id"]]
        refs = fetch_references(ids, api_key=api_key)
        links = find_citation_links(refs, seeds)
        if links:
            store.save_citations([(c, cited) for c, cs in links.items() for cited in cs])
    return dict.fromkeys(links, 1.0), {
        "n_seeds": len(seeds),
        "ids_requested": len(ids),
        "papers_with_refs": len(refs),
        "ref_edges": sum(len(v) for v in refs.values()),
    }
