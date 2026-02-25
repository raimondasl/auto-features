"""Tests for reporadar.feedback."""

from __future__ import annotations

from reporadar.feedback import compute_adjusted_weights, find_similar_to_highly_rated


class TestComputeAdjustedWeights:
    def test_returns_dict_with_same_keys(self) -> None:
        rated = [
            {"rating": 5, "keyword_score": 0.8, "category_score": 0.5, "recency_score": 0.3},
            {"rating": 5, "keyword_score": 0.9, "category_score": 0.6, "recency_score": 0.2},
            {"rating": 1, "keyword_score": 0.1, "category_score": 0.8, "recency_score": 0.9},
            {"rating": 1, "keyword_score": 0.2, "category_score": 0.7, "recency_score": 0.8},
        ]
        current = {"w_keyword": 1.0, "w_category": 0.5, "w_recency": 0.3}
        result = compute_adjusted_weights(rated, current, learning_rate=0.1)
        assert set(result.keys()) == set(current.keys())

    def test_weights_are_positive(self) -> None:
        rated = [
            {"rating": 5, "keyword_score": 0.9, "category_score": 0.1, "recency_score": 0.1},
            {"rating": 5, "keyword_score": 0.8, "category_score": 0.2, "recency_score": 0.1},
            {"rating": 1, "keyword_score": 0.1, "category_score": 0.9, "recency_score": 0.9},
            {"rating": 1, "keyword_score": 0.2, "category_score": 0.8, "recency_score": 0.8},
        ]
        current = {"w_keyword": 1.0, "w_category": 0.5, "w_recency": 0.3}
        result = compute_adjusted_weights(rated, current, learning_rate=0.5)
        for v in result.values():
            assert v >= 0

    def test_insufficient_data_returns_current(self) -> None:
        # Only one rating → can't fit logistic regression
        rated = [
            {"rating": 5, "keyword_score": 0.8, "category_score": 0.5, "recency_score": 0.3},
        ]
        current = {"w_keyword": 1.0, "w_category": 0.5, "w_recency": 0.3}
        result = compute_adjusted_weights(rated, current)
        assert result == current

    def test_all_same_class_returns_current(self) -> None:
        # All positive, no negative → can't fit
        rated = [
            {"rating": 5, "keyword_score": 0.8, "category_score": 0.5, "recency_score": 0.3},
            {"rating": 4, "keyword_score": 0.7, "category_score": 0.6, "recency_score": 0.2},
        ]
        current = {"w_keyword": 1.0, "w_category": 0.5, "w_recency": 0.3}
        result = compute_adjusted_weights(rated, current)
        assert result == current

    def test_neutral_ratings_ignored(self) -> None:
        # Rating=3 should be skipped
        rated = [
            {"rating": 3, "keyword_score": 0.5, "category_score": 0.5, "recency_score": 0.5},
            {"rating": 3, "keyword_score": 0.5, "category_score": 0.5, "recency_score": 0.5},
        ]
        current = {"w_keyword": 1.0, "w_category": 0.5, "w_recency": 0.3}
        result = compute_adjusted_weights(rated, current)
        assert result == current

    def test_learning_rate_zero_preserves_current(self) -> None:
        rated = [
            {"rating": 5, "keyword_score": 0.9, "category_score": 0.1, "recency_score": 0.1},
            {"rating": 1, "keyword_score": 0.1, "category_score": 0.9, "recency_score": 0.9},
        ]
        current = {"w_keyword": 1.0, "w_category": 0.5, "w_recency": 0.3}
        result = compute_adjusted_weights(rated, current, learning_rate=0.0)
        # With lr=0, should preserve exactly current weights
        for k in current:
            assert abs(result[k] - current[k]) < 0.0001


class TestFindSimilarToHighlyRated:
    def test_finds_similar_papers(self) -> None:
        papers = [
            {
                "arxiv_id": "c1",
                "title": "Transformer attention mechanism",
                "abstract": "We propose a transformer with attention.",
                "score_total": 0.7,
                "url": "http://test/c1",
            },
            {
                "arxiv_id": "c2",
                "title": "Quantum computing basics",
                "abstract": "A review of quantum computing.",
                "score_total": 0.5,
                "url": "http://test/c2",
            },
        ]
        rated = {
            "r1": {
                "arxiv_id": "r1",
                "title": "Attention is all you need",
                "abstract": "A transformer model using attention for sequences.",
                "rating": 5,
            },
        }
        result = find_similar_to_highly_rated(papers, rated)
        # c1 is more similar to the rated paper than c2
        assert len(result) >= 1
        assert result[0]["arxiv_id"] == "c1"

    def test_excludes_already_rated(self) -> None:
        papers = [
            {
                "arxiv_id": "r1",
                "title": "Same paper",
                "abstract": "Same abstract",
                "score_total": 0.9,
                "url": "http://test/r1",
            },
        ]
        rated = {
            "r1": {
                "arxiv_id": "r1",
                "title": "Same paper",
                "abstract": "Same abstract",
                "rating": 5,
            },
        }
        result = find_similar_to_highly_rated(papers, rated)
        assert len(result) == 0

    def test_no_highly_rated(self) -> None:
        papers = [
            {"arxiv_id": "c1", "title": "A paper", "abstract": "Text", "score_total": 0.5},
        ]
        rated = {
            "r1": {"arxiv_id": "r1", "title": "Paper", "abstract": "Text", "rating": 2},
        }
        result = find_similar_to_highly_rated(papers, rated)
        assert result == []

    def test_respects_top_k(self) -> None:
        papers = [
            {
                "arxiv_id": f"c{i}",
                "title": "Transformer model",
                "abstract": "Attention based transformer.",
                "score_total": 0.5,
                "url": f"http://test/c{i}",
            }
            for i in range(10)
        ]
        rated = {
            "r1": {
                "arxiv_id": "r1",
                "title": "Transformer paper",
                "abstract": "Attention transformer.",
                "rating": 5,
            },
        }
        result = find_similar_to_highly_rated(papers, rated, top_k=2)
        assert len(result) <= 2

    def test_empty_inputs(self) -> None:
        assert find_similar_to_highly_rated([], {}) == []
        assert find_similar_to_highly_rated([], {"r1": {"rating": 5}}) == []
