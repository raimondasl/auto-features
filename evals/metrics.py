"""Ranking-quality metrics for the RepoRadar eval benchmark.

All functions take ``ranked_labels`` — a list of binary relevance labels (1 =
relevant/gold, 0 = distractor) in *ranked order* (best-scored first). This is
the standard information-retrieval setup used by every ranking-metric guide.
"""

from __future__ import annotations

import math


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
