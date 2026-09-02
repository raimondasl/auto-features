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


def average_ranks(values: list[float]) -> list[float]:
    """Ranks with ties sharing their average rank.

    Both consumers are tie-heavy and would be biased by position-breaking: net@2 puts many
    cases at exactly 0.0 (the thin-docs correlation), and a judge's ordinal rubric score has
    four levels for hundreds of papers (the adoption AUC).
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def roc_auc(positives: list[float], controls: list[float]) -> float:
    """P(a random positive outranks a random control), ties counted as half.

    The Mann-Whitney form, so ties are handled by the shared ranks rather than by a
    threshold. **Level-free by construction**: adding a constant to every score, or moving a
    judge's bar, leaves it unchanged. That is exactly why the frame makes it primary --
    NR-59 measured the two judges ordering alike (AUCs 0.027 apart) while disagreeing about
    level by 0.380, so any statistic evaluated at each judge's own threshold restates the
    level disagreement instead of measuring discrimination.
    """
    n1, n2 = len(positives), len(controls)
    if not n1 or not n2:
        return float("nan")
    ranks = average_ranks(list(positives) + list(controls))
    return (sum(ranks[:n1]) - n1 * (n1 + 1) / 2) / (n1 * n2)
