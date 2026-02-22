"""Heuristic scoring of papers against a repo profile."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from reporadar.config import QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenization of text into word-level tokens."""
    return set(re.findall(r"[a-z][a-z0-9_-]+", text.lower()))


def score_keyword_overlap(
    paper: dict[str, Any],
    profile: RepoProfile,
) -> float:
    """Score based on keyword overlap between paper title+abstract and profile.

    Returns a score in [0, 1]. Each matching profile keyword contributes its
    TF-IDF weight. The result is normalized by the sum of all profile weights.
    """
    if not profile.keywords:
        return 0.0

    paper_tokens = _tokenize(paper["title"] + " " + paper["abstract"])

    matched_weight = 0.0
    total_weight = 0.0
    for term, weight in profile.keywords:
        total_weight += weight
        # Check if any token in the term matches paper tokens
        term_tokens = _tokenize(term)
        if term_tokens & paper_tokens:
            matched_weight += weight

    if total_weight == 0:
        return 0.0

    return min(matched_weight / total_weight, 1.0)


def score_category_match(
    paper: dict[str, Any],
    target_categories: list[str],
) -> float:
    """Score based on how many of the paper's categories match the target list.

    Returns a score in [0, 1].
    """
    if not target_categories or not paper.get("categories"):
        return 0.0

    target_set = set(target_categories)
    paper_cats = set(paper["categories"])

    overlap = len(target_set & paper_cats)
    return min(overlap / len(target_set), 1.0)


def score_recency(paper: dict[str, Any], lookback_days: int = 14) -> float:
    """Score based on how recent the paper is.

    Returns 1.0 for papers published today, decaying linearly to 0.0
    at *lookback_days* ago. Papers older than the lookback window get 0.0.
    """
    try:
        published = datetime.fromisoformat(paper["published"])
    except (ValueError, KeyError):
        return 0.0

    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)

    now = datetime.now(UTC)
    age_days = (now - published).total_seconds() / 86400

    if age_days < 0:
        return 1.0
    if age_days >= lookback_days:
        return 0.0

    return 1.0 - (age_days / lookback_days)


def compute_exclude_penalty(
    paper: dict[str, Any],
    exclude_terms: list[str],
) -> float:
    """Compute a penalty multiplier for papers matching exclude terms.

    Returns a value in (0, 1]. Each matched exclude term multiplies the
    score by 0.5, so papers matching many exclude terms get heavily penalized.
    """
    if not exclude_terms:
        return 1.0

    paper_tokens = _tokenize(paper["title"] + " " + paper["abstract"])
    penalty = 1.0
    for term in exclude_terms:
        term_tokens = _tokenize(term)
        if term_tokens & paper_tokens:
            penalty *= 0.5

    return penalty


def score_paper(
    paper: dict[str, Any],
    profile: RepoProfile,
    ranking_cfg: RankingConfig,
    queries_cfg: QueriesConfig,
    arxiv_categories: list[str],
    lookback_days: int = 14,
) -> dict[str, Any]:
    """Compute a combined score for a single paper.

    Returns a score dict suitable for PaperStore.save_scores().
    """
    kw = score_keyword_overlap(paper, profile)
    cat = score_category_match(paper, arxiv_categories)
    rec = score_recency(paper, lookback_days)

    raw_total = (
        ranking_cfg.w_keyword * kw + ranking_cfg.w_category * cat + ranking_cfg.w_recency * rec
    )

    penalty = compute_exclude_penalty(paper, queries_cfg.exclude)
    total = raw_total * penalty

    return {
        "arxiv_id": paper["arxiv_id"],
        "score_total": round(total, 4),
        "keyword_score": round(kw, 4),
        "category_score": round(cat, 4),
        "recency_score": round(rec, 4),
        "matched_query": paper.get("matched_query"),
    }


def rank_papers(
    papers: list[dict[str, Any]],
    profile: RepoProfile,
    ranking_cfg: RankingConfig,
    queries_cfg: QueriesConfig,
    arxiv_categories: list[str],
    lookback_days: int = 14,
) -> list[dict[str, Any]]:
    """Score and rank a list of papers. Returns score dicts sorted by score descending."""
    scores = [
        score_paper(paper, profile, ranking_cfg, queries_cfg, arxiv_categories, lookback_days)
        for paper in papers
    ]
    scores.sort(key=lambda s: s["score_total"], reverse=True)
    return scores
