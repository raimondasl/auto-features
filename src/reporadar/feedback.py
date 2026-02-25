"""User feedback loop — adjust ranking weights from paper ratings."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_adjusted_weights(
    rated_scores: list[dict[str, Any]],
    current_weights: dict[str, float],
    learning_rate: float = 0.1,
) -> dict[str, float]:
    """Blend learned coefficients from rated papers with current weights.

    *rated_scores* should come from ``store.get_rated_paper_scores()``.
    Each entry must have ``rating`` (1-5) and the component score fields.

    Uses logistic regression on rated papers (rating >= 4 → positive,
    rating <= 2 → negative, 3 ignored) to learn which score components
    best predict user preference. The learned coefficients are blended
    with *current_weights* at *learning_rate*.

    Returns a new weights dict with the same keys as *current_weights*.
    """
    features = ["keyword_score", "category_score", "recency_score"]
    has_embedding = any(r.get("embedding_score") is not None for r in rated_scores)
    has_citation = any(r.get("citation_score") is not None for r in rated_scores)
    if has_embedding:
        features.append("embedding_score")
    if has_citation:
        features.append("citation_score")

    weight_keys = {
        "keyword_score": "w_keyword",
        "category_score": "w_category",
        "recency_score": "w_recency",
        "embedding_score": "w_embedding",
        "citation_score": "w_citations",
    }

    # Build training data: positive (rating >= 4), negative (rating <= 2)
    X_rows: list[list[float]] = []
    y_labels: list[int] = []
    for entry in rated_scores:
        rating = entry["rating"]
        if rating >= 4:
            label = 1
        elif rating <= 2:
            label = 0
        else:
            continue  # skip neutral ratings

        row = [float(entry.get(f) or 0.0) for f in features]
        X_rows.append(row)
        y_labels.append(label)

    if len(X_rows) < 2 or len(set(y_labels)) < 2:
        return dict(current_weights)

    X = np.array(X_rows)
    y = np.array(y_labels)

    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X, y)

    # Blend learned coefficients with current weights
    learned_coeffs = clf.coef_[0]
    # Normalize learned coefficients to be non-negative and sum-like
    abs_coeffs = np.abs(learned_coeffs)
    if abs_coeffs.sum() > 0:
        normalized = abs_coeffs / abs_coeffs.sum()
    else:
        return dict(current_weights)

    # Scale to match current weight magnitudes
    current_sum = sum(current_weights.get(weight_keys[f], 0.0) for f in features)
    if current_sum == 0:
        current_sum = 1.0

    new_weights = dict(current_weights)
    for i, feat in enumerate(features):
        wk = weight_keys[feat]
        if wk in new_weights:
            current_val = new_weights[wk]
            learned_val = normalized[i] * current_sum
            new_weights[wk] = round(
                current_val * (1 - learning_rate) + learned_val * learning_rate, 4
            )

    return new_weights


def find_similar_to_highly_rated(
    papers: list[dict[str, Any]],
    rated_papers: dict[str, dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Find papers similar to highly-rated ones by keyword overlap.

    *rated_papers* is {arxiv_id: paper_dict_with_rating}.
    Returns up to *top_k* papers not already rated, sorted by similarity.
    """
    # Collect tokens from highly-rated papers (rating >= 4)
    positive_tokens: set[str] = set()
    for arxiv_id, rp in rated_papers.items():
        if rp.get("rating", 0) >= 4:
            text = (rp.get("title", "") + " " + rp.get("abstract", "")).lower()
            tokens = set(re.findall(r"[a-z][a-z0-9_-]+", text))
            positive_tokens |= tokens

    if not positive_tokens:
        return []

    # Common English stop words to filter out
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "been", "have", "has", "had", "will", "would", "can", "could",
        "which", "their", "our", "its", "also", "than", "into", "these",
        "those", "such", "when", "where", "each", "both", "more", "most",
        "some", "other", "about", "between", "through", "during", "before",
        "after", "above", "below", "not", "but", "they", "them", "then",
        "what", "how", "all", "any", "over", "only",
    }
    positive_tokens -= stop_words

    # Score candidate papers by overlap with positive tokens
    rated_ids = set(rated_papers.keys())
    scored: list[tuple[float, dict[str, Any]]] = []

    for paper in papers:
        if paper["arxiv_id"] in rated_ids:
            continue
        text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        paper_tokens = set(re.findall(r"[a-z][a-z0-9_-]+", text))
        paper_tokens -= stop_words
        if not paper_tokens:
            continue
        overlap = len(positive_tokens & paper_tokens) / len(positive_tokens)
        if overlap > 0:
            scored.append((overlap, paper))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]
