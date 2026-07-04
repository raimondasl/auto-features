"""Ranking-quality metrics for the RepoRadar eval benchmark.

All functions take ``ranked_labels`` — a list of binary relevance labels (1 =
relevant/gold, 0 = distractor) in *ranked order* (best-scored first). This is
the standard information-retrieval setup used by every ranking-metric guide.
"""

from __future__ import annotations

import math
from typing import Any


def precision_at_k(ranked_labels: list[int], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top = ranked_labels[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def recall_at_k(ranked_labels: list[int], k: int, total_relevant: int | None = None) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    total = total_relevant if total_relevant is not None else sum(ranked_labels)
    if total <= 0:
        return 0.0
    return sum(ranked_labels[:k]) / total


def dcg_at_k(ranked_labels: list[int], k: int) -> float:
    """Discounted cumulative gain with binary gains and log2 position discount."""
    dcg = 0.0
    for i, rel in enumerate(ranked_labels[:k], start=1):
        if rel:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_labels: list[int], k: int) -> float:
    """DCG@k normalized by the ideal DCG@k (all relevant items ranked first)."""
    ideal = sorted(ranked_labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_labels, k) / idcg


def mrr(ranked_labels: list[int]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none)."""
    for i, rel in enumerate(ranked_labels, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked_labels: list[int]) -> float:
    """Mean of precision@i taken at each rank position that holds a relevant item."""
    total_relevant = sum(ranked_labels)
    if total_relevant == 0:
        return 0.0
    hits = 0
    running = 0.0
    for i, rel in enumerate(ranked_labels, start=1):
        if rel:
            hits += 1
            running += hits / i
    return running / total_relevant


def separation(gold_scores: list[float], distractor_scores: list[float]) -> float:
    """Mean gold score minus mean distractor score.

    Positive means the ranker scores relevant papers higher on average — the
    single most interpretable number for "does it tell signal from noise?".
    """
    if not gold_scores or not distractor_scores:
        return 0.0
    return (sum(gold_scores) / len(gold_scores)) - (sum(distractor_scores) / len(distractor_scores))


def evaluate_ranking(
    ranked: list[dict],
    labels_by_id: dict[str, int],
    k: int,
) -> dict[str, float]:
    """Compute the full metric suite for one ranked result list.

    *ranked* is the score dicts sorted best-first (each has ``arxiv_id`` and
    ``score_total``). *labels_by_id* maps arxiv_id -> 1 (gold) / 0 (distractor).
    """
    ranked_labels = [labels_by_id.get(r["arxiv_id"], 0) for r in ranked]
    gold_scores = [r["score_total"] for r in ranked if labels_by_id.get(r["arxiv_id"], 0) == 1]
    distractor_scores = [
        r["score_total"] for r in ranked if labels_by_id.get(r["arxiv_id"], 0) == 0
    ]
    total_relevant = sum(labels_by_id.values())

    return {
        "precision@k": precision_at_k(ranked_labels, k),
        "recall@k": recall_at_k(ranked_labels, k, total_relevant),
        "ndcg@k": ndcg_at_k(ranked_labels, k),
        "mrr": mrr(ranked_labels),
        "map": average_precision(ranked_labels),
        "separation": separation(gold_scores, distractor_scores),
        "n_gold": float(total_relevant),
        "n_candidates": float(len(ranked)),
    }


# ── Tier B: judge-based, abstention-aware metrics ──────────────────────────
#
# These operate on a system's *returned* papers, each carrying a judge score in
# 0..3 (2+ = "genuinely actionable"). They reward precision and correct
# abstention and penalize false positives, per the design in README.md.

RELEVANT_THRESHOLD = 2  # judge score >= this counts as genuinely actionable


def judged_precision(returned_scores: list[int], threshold: int = RELEVANT_THRESHOLD) -> float:
    """Fraction of returned papers that are genuinely actionable.

    Returns NaN for an empty list — an empty result is an *abstention*, not a
    precision-0 failure, so it must be excluded from precision averages and
    judged by ``net_actionable_value`` / ``abstained`` instead.
    """
    if not returned_scores:
        return float("nan")
    return sum(1 for s in returned_scores if s >= threshold) / len(returned_scores)


def net_actionable_value(
    returned_scores: list[int],
    lam: float = 2.0,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """(# genuinely actionable) − lam · (# non-actionable) over returned papers.

    With lam > 1 a junk paper costs more than a good paper earns, so returning
    nothing (value 0) beats returning noise. This is the headline Tier B metric.
    """
    good = sum(1 for s in returned_scores if s >= threshold)
    bad = sum(1 for s in returned_scores if s < threshold)
    return good - lam * bad


def graded_dcg(gains: list[int], k: int) -> float:
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        if g:
            dcg += (2**g - 1) / math.log2(i + 1)
    return dcg


def graded_ndcg(ranked_gains: list[int], ideal_gains: list[int], k: int) -> float:
    """nDCG@k with graded (0..3) relevance.

    *ranked_gains* are the judge scores of a system's returned papers in its own
    ranked order; *ideal_gains* are all judge scores in the pool, best-first.
    """
    idcg = graded_dcg(sorted(ideal_gains, reverse=True), k)
    if idcg == 0.0:
        return 0.0
    return graded_dcg(ranked_gains, k) / idcg


def summarize_system(
    returned_scores: list[int],
    pool_gains: list[int],
    *,
    k: int = 10,
    n_hallucinated: int = 0,
    lambdas: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, Any]:
    """Full Tier B metric bundle for one system on one repo.

    ``pool_has_relevant`` drives abstention scoring: when the pool contains no
    actionable papers, the correct behavior is to return nothing.
    """
    pool_has_relevant = any(g >= RELEVANT_THRESHOLD for g in pool_gains)
    n_returned = len(returned_scores)
    n_good = sum(1 for s in returned_scores if s >= RELEVANT_THRESHOLD)

    result: dict[str, Any] = {
        "n_returned": n_returned,
        "n_actionable": n_good,
        "n_hallucinated": n_hallucinated,
        "precision": judged_precision(returned_scores),
        "ndcg@k": graded_ndcg(returned_scores, pool_gains, k),
        "abstained": n_returned == 0,
        "pool_has_relevant": pool_has_relevant,
    }
    for lam in lambdas:
        result[f"net_value@{lam:g}"] = net_actionable_value(returned_scores, lam)
    # Correct abstention: when nothing in the pool is actionable, returning
    # nothing is a win; each returned (non-actionable) paper is a mistake.
    if not pool_has_relevant:
        result["abstention_correct"] = 1.0 if n_returned == 0 else 0.0
    return result
