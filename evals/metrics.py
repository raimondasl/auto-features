"""Benchmark-only metrics for the ``evals/`` harness.

The shared IR primitives now live in :mod:`reporadar.metrics` so ``rr eval`` and this
benchmark cannot drift apart; they are re-exported here so the runner scripts keep
importing them from one place. What remains below is Tier-B-specific: judge-based,
abstention-aware scoring that has no meaning outside the benchmark.
"""

from __future__ import annotations

import math
from typing import Any

from reporadar.metrics import (
    average_precision,
    dcg_at_k,
    evaluate_ranking,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    separation,
)

__all__ = [
    "RELEVANT_THRESHOLD",
    "average_precision",
    "dcg_at_k",
    "evaluate_ranking",
    "graded_dcg",
    "graded_ndcg",
    "judged_precision",
    "mrr",
    "ndcg_at_k",
    "net_actionable_value",
    "precision_at_k",
    "recall_at_k",
    "separation",
    "summarize_system",
]


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
