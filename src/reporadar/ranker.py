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
    category_weights: dict[str, float] | None = None,
) -> float:
    """Score based on how many of the paper's categories match the target list.

    When *category_weights* is provided and non-empty, each matching category
    contributes its weight (unmentioned categories default to 1.0). The result
    is normalized by the sum of all target category weights.

    Returns a score in [0, 1].
    """
    if not target_categories or not paper.get("categories"):
        return 0.0

    paper_cats = set(paper["categories"])

    if category_weights:
        matched_weight = 0.0
        total_weight = 0.0
        for cat in target_categories:
            w = category_weights.get(cat, 1.0)
            total_weight += w
            if cat in paper_cats:
                matched_weight += w
        if total_weight == 0:
            return 0.0
        return min(matched_weight / total_weight, 1.0)

    target_set = set(target_categories)
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
    embedding_score: float | None = None,
    citation_score: float | None = None,
) -> dict[str, Any]:
    """Compute a combined score for a single paper.

    Returns a score dict suitable for PaperStore.save_scores().
    """
    kw = score_keyword_overlap(paper, profile)
    cat = score_category_match(
        paper, arxiv_categories, category_weights=ranking_cfg.category_weights or None
    )
    rec = score_recency(paper, lookback_days)

    raw_total = (
        ranking_cfg.w_keyword * kw + ranking_cfg.w_category * cat + ranking_cfg.w_recency * rec
    )
    weight_sum = ranking_cfg.w_keyword + ranking_cfg.w_category + ranking_cfg.w_recency

    w_embedding = getattr(ranking_cfg, "w_embedding", 0.0)
    if embedding_score is not None and w_embedding > 0:
        raw_total += w_embedding * embedding_score
        weight_sum += w_embedding

    w_citations = getattr(ranking_cfg, "w_citations", 0.0)
    if citation_score is not None and w_citations > 0:
        raw_total += w_citations * citation_score
        weight_sum += w_citations

    normalized = raw_total / weight_sum if weight_sum > 0 else 0.0

    penalty = compute_exclude_penalty(paper, queries_cfg.exclude)
    total = normalized * penalty

    return {
        "arxiv_id": paper["arxiv_id"],
        "score_total": round(total, 4),
        "keyword_score": round(kw, 4),
        "category_score": round(cat, 4),
        "recency_score": round(rec, 4),
        "embedding_score": round(embedding_score, 4) if embedding_score is not None else None,
        "citation_score": round(citation_score, 4) if citation_score is not None else None,
        "matched_query": paper.get("matched_query"),
    }


def format_score_explanation(score_dict: dict[str, Any], ranking_cfg: RankingConfig) -> str:
    """Return a multi-line string showing weight * component = contribution for each component."""
    lines = [f"  Paper: {score_dict['arxiv_id']}"]
    components = [
        ("keyword", "keyword_score", ranking_cfg.w_keyword),
        ("category", "category_score", ranking_cfg.w_category),
        ("recency", "recency_score", ranking_cfg.w_recency),
    ]
    if "embedding_score" in score_dict and score_dict["embedding_score"] is not None:
        components.append(("embedding", "embedding_score", getattr(ranking_cfg, "w_embedding", 0)))
    if "citation_score" in score_dict and score_dict["citation_score"] is not None:
        components.append(("citation", "citation_score", getattr(ranking_cfg, "w_citations", 0)))

    for name, key, weight in components:
        val = score_dict.get(key, 0) or 0
        contribution = weight * val
        lines.append(f"    {name:12s}: {weight:.2f} * {val:.4f} = {contribution:.4f}")

    lines.append(f"    {'total':12s}: {score_dict['score_total']:.4f}")
    return "\n".join(lines)


def score_distribution(scores: list[dict[str, Any]]) -> dict[str, float]:
    """Compute distribution stats from a list of score dicts.

    Returns {"mean": ..., "median": ..., "min": ..., "max": ..., "count": ...}.
    """
    if not scores:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    totals = sorted(s["score_total"] for s in scores)
    n = len(totals)
    mean = sum(totals) / n
    median = totals[n // 2] if n % 2 == 1 else (totals[n // 2 - 1] + totals[n // 2]) / 2

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": round(totals[0], 4),
        "max": round(totals[-1], 4),
        "count": n,
    }


def rank_papers(
    papers: list[dict[str, Any]],
    profile: RepoProfile,
    ranking_cfg: RankingConfig,
    queries_cfg: QueriesConfig,
    arxiv_categories: list[str],
    lookback_days: int = 14,
    repo_embedding: Any = None,
    citation_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score and rank a list of papers. Returns score dicts sorted by score descending."""
    scores = []
    for paper in papers:
        emb_score = None
        if repo_embedding is not None:
            try:
                from reporadar.embeddings import compute_paper_embedding, cosine_similarity

                paper_emb = compute_paper_embedding(paper)
                emb_score = max(0.0, cosine_similarity(repo_embedding, paper_emb))
            except (RuntimeError, ImportError):
                pass

        cit_score = None
        if citation_scores is not None:
            cit_score = citation_scores.get(paper["arxiv_id"])

        scores.append(
            score_paper(
                paper,
                profile,
                ranking_cfg,
                queries_cfg,
                arxiv_categories,
                lookback_days,
                embedding_score=emb_score,
                citation_score=cit_score,
            )
        )
    scores.sort(key=lambda s: s["score_total"], reverse=True)
    return scores
