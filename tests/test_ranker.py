"""Tests for reporadar.ranker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from reporadar.config import QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile
from reporadar.ranker import (
    compute_exclude_penalty,
    format_score_explanation,
    rank_papers,
    score_category_match,
    score_distribution,
    score_keyword_overlap,
    score_paper,
    score_recency,
)


def _make_paper(**overrides) -> dict:
    now = datetime.now(UTC)
    base = {
        "arxiv_id": "2401.12345v1",
        "title": "Retrieval Augmented Generation with Long Context Transformers",
        "authors": ["Alice Smith"],
        "abstract": "We propose a novel retrieval augmented generation framework "
        "that leverages long context transformers for improved question answering.",
        "categories": ["cs.CL", "cs.LG"],
        "published": now.isoformat(),
        "url": "http://arxiv.org/abs/2401.12345v1",
    }
    base.update(overrides)
    return base


def _make_profile(**overrides) -> RepoProfile:
    defaults = {
        "keywords": [
            ("retrieval", 0.8),
            ("transformers", 0.7),
            ("generation", 0.6),
            ("embeddings", 0.4),
        ],
        "anchors": ["torch", "transformers"],
        "domains": ["deep learning", "NLP"],
    }
    defaults.update(overrides)
    return RepoProfile(**defaults)


class TestScoreKeywordOverlap:
    def test_full_overlap(self) -> None:
        paper = _make_paper()
        profile = _make_profile(keywords=[("retrieval", 0.5), ("transformers", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        paper = _make_paper()
        profile = _make_profile(keywords=[("retrieval", 0.5), ("quantum", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert 0.4 < score < 0.6

    def test_no_overlap(self) -> None:
        paper = _make_paper(
            title="Quantum Computing Advances",
            abstract="A new quantum error correction code.",
        )
        profile = _make_profile(keywords=[("retrieval", 0.5), ("transformers", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert score == 0.0

    def test_empty_profile(self) -> None:
        paper = _make_paper()
        profile = _make_profile(keywords=[])
        assert score_keyword_overlap(paper, profile) == 0.0

    def test_bigram_match(self) -> None:
        # Terms like "long context" should match if individual tokens match
        paper = _make_paper(title="Long Context Window Models")
        profile = _make_profile(keywords=[("long", 0.5), ("context", 0.5)])
        score = score_keyword_overlap(paper, profile)
        assert score == pytest.approx(1.0)


class TestScoreCategoryMatch:
    def test_full_match(self) -> None:
        paper = _make_paper(categories=["cs.CL", "cs.LG"])
        score = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        score = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score == pytest.approx(0.5)

    def test_no_match(self) -> None:
        paper = _make_paper(categories=["cs.CV"])
        score = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score == 0.0

    def test_empty_targets(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        assert score_category_match(paper, []) == 0.0

    def test_empty_paper_categories(self) -> None:
        paper = _make_paper(categories=[])
        assert score_category_match(paper, ["cs.CL"]) == 0.0


class TestScoreRecency:
    def test_today_is_1(self) -> None:
        now = datetime.now(UTC)
        paper = _make_paper(published=now.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score > 0.95

    def test_old_paper_is_0(self) -> None:
        old = datetime(2020, 1, 1, tzinfo=UTC)
        paper = _make_paper(published=old.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_midpoint(self) -> None:
        mid = datetime.now(UTC) - timedelta(days=7)
        paper = _make_paper(published=mid.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert 0.4 < score < 0.6

    def test_decays_linearly(self) -> None:
        now = datetime.now(UTC)
        scores = []
        for days_ago in [1, 5, 10, 13]:
            pub = now - timedelta(days=days_ago)
            paper = _make_paper(published=pub.isoformat())
            scores.append(score_recency(paper, lookback_days=14))
        # Should be monotonically decreasing
        assert scores == sorted(scores, reverse=True)


class TestExcludePenalty:
    def test_no_exclude(self) -> None:
        paper = _make_paper()
        assert compute_exclude_penalty(paper, []) == 1.0

    def test_single_match(self) -> None:
        paper = _make_paper(title="A Survey of RAG Methods")
        penalty = compute_exclude_penalty(paper, ["survey"])
        assert penalty == pytest.approx(0.5)

    def test_multiple_matches(self) -> None:
        paper = _make_paper(
            title="A Survey and Benchmark of RAG Methods",
            abstract="We benchmark various survey approaches.",
        )
        penalty = compute_exclude_penalty(paper, ["survey", "benchmark"])
        assert penalty == pytest.approx(0.25)

    def test_no_match(self) -> None:
        paper = _make_paper()
        penalty = compute_exclude_penalty(paper, ["quantum", "biology"])
        assert penalty == 1.0


class TestScorePaper:
    def test_returns_expected_keys(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert "arxiv_id" in result
        assert "score_total" in result
        assert "keyword_score" in result
        assert "category_score" in result
        assert "recency_score" in result

    def test_weights_affect_total(self) -> None:
        paper = _make_paper()
        profile = _make_profile()

        # Heavy keyword weight
        kw_heavy = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=10.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        # Heavy recency weight
        rec_heavy = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=0.0, w_category=0.0, w_recency=10.0),
            QueriesConfig(),
            ["cs.CL"],
        )

        # With different weights, totals should differ
        assert kw_heavy["score_total"] != rec_heavy["score_total"]

    def test_exclude_reduces_total(self) -> None:
        paper = _make_paper(title="A Survey of RAG")
        profile = _make_profile()

        without_exclude = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
        )
        with_exclude = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(exclude=["survey"]),
            ["cs.CL"],
        )

        assert with_exclude["score_total"] < without_exclude["score_total"]


class TestEdgeCases:
    def test_recency_with_invalid_date(self) -> None:
        paper = _make_paper(published="not-a-date")
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_recency_with_missing_published(self) -> None:
        paper = _make_paper()
        del paper["published"]
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_recency_future_date(self) -> None:
        future = datetime.now(UTC) + timedelta(days=5)
        paper = _make_paper(published=future.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score == 1.0

    def test_keyword_overlap_empty_abstract(self) -> None:
        paper = _make_paper(title="", abstract="")
        profile = _make_profile()
        score = score_keyword_overlap(paper, profile)
        assert score == 0.0

    def test_all_weights_zero(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=0.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        assert result["score_total"] == 0.0

    def test_category_superset(self) -> None:
        # Paper has more categories than target — should still cap at 1.0
        paper = _make_paper(categories=["cs.CL", "cs.LG", "cs.AI"])
        score = score_category_match(paper, ["cs.CL"])
        assert score == pytest.approx(1.0)


class TestScoreNormalization:
    def test_default_weights_normalized(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert result["score_total"] <= 1.0

    def test_all_max_scores_equals_one(self) -> None:
        # Paper matching all keywords, all categories, published now → 1.0
        now = datetime.now(UTC)
        paper = _make_paper(
            title="retrieval transformers generation embeddings",
            abstract="retrieval transformers generation embeddings",
            categories=["cs.CL", "cs.LG"],
            published=now.isoformat(),
        )
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert result["score_total"] == pytest.approx(1.0, abs=0.02)

    def test_custom_weights_still_normalized(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=5.0, w_category=3.0, w_recency=2.0),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
        )
        assert result["score_total"] <= 1.0

    def test_zero_weights_returns_zero(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=0.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        assert result["score_total"] == 0.0


class TestRankPapers:
    def test_returns_sorted_by_score(self) -> None:
        profile = _make_profile()
        papers = [
            _make_paper(
                arxiv_id="2401.00001v1",
                title="Quantum Computing Basics",
                abstract="No overlap with profile.",
            ),
            _make_paper(
                arxiv_id="2401.00002v1",
                title="Retrieval Augmented Generation",
                abstract="Transformers for retrieval and generation.",
            ),
        ]

        scores = rank_papers(
            papers,
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
        )

        assert len(scores) == 2
        assert scores[0]["score_total"] >= scores[1]["score_total"]
        # The RAG paper should rank higher
        assert scores[0]["arxiv_id"] == "2401.00002v1"

    def test_empty_papers(self) -> None:
        profile = _make_profile()
        scores = rank_papers(
            [],
            profile,
            RankingConfig(),
            QueriesConfig(),
            ["cs.CL"],
        )
        assert scores == []


class TestFormatScoreExplanation:
    def test_contains_component_names(self) -> None:
        score_dict = {
            "arxiv_id": "2401.00001v1",
            "score_total": 0.75,
            "keyword_score": 0.8,
            "category_score": 0.6,
            "recency_score": 0.9,
        }
        result = format_score_explanation(score_dict, RankingConfig())
        assert "keyword" in result
        assert "category" in result
        assert "recency" in result
        assert "2401.00001v1" in result

    def test_contains_weight_values(self) -> None:
        score_dict = {
            "arxiv_id": "2401.00001v1",
            "score_total": 0.75,
            "keyword_score": 0.8,
            "category_score": 0.6,
            "recency_score": 0.9,
        }
        result = format_score_explanation(score_dict, RankingConfig(w_keyword=2.0))
        assert "2.00" in result

    def test_contains_total(self) -> None:
        score_dict = {
            "arxiv_id": "2401.00001v1",
            "score_total": 0.75,
            "keyword_score": 0.8,
            "category_score": 0.6,
            "recency_score": 0.9,
        }
        result = format_score_explanation(score_dict, RankingConfig())
        assert "total" in result
        assert "0.7500" in result


class TestScoreDistribution:
    def test_known_inputs(self) -> None:
        scores = [
            {"score_total": 0.2},
            {"score_total": 0.4},
            {"score_total": 0.6},
            {"score_total": 0.8},
        ]
        dist = score_distribution(scores)
        assert dist["mean"] == pytest.approx(0.5)
        assert dist["median"] == pytest.approx(0.5)
        assert dist["min"] == pytest.approx(0.2)
        assert dist["max"] == pytest.approx(0.8)
        assert dist["count"] == 4

    def test_empty_list(self) -> None:
        dist = score_distribution([])
        assert dist["mean"] == 0.0
        assert dist["median"] == 0.0
        assert dist["min"] == 0.0
        assert dist["max"] == 0.0
        assert dist["count"] == 0

    def test_single_item(self) -> None:
        dist = score_distribution([{"score_total": 0.5}])
        assert dist["mean"] == pytest.approx(0.5)
        assert dist["median"] == pytest.approx(0.5)
        assert dist["count"] == 1


class TestEmbeddingScoreIntegration:
    def test_embedding_score_included_when_provided(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3, w_embedding=1.5),
            QueriesConfig(),
            ["cs.CL", "cs.LG"],
            embedding_score=0.8,
        )
        assert result["embedding_score"] == pytest.approx(0.8)
        assert result["score_total"] > 0

    def test_embedding_score_ignored_when_weight_zero(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        without = score_paper(
            paper,
            profile,
            RankingConfig(w_embedding=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        with_emb = score_paper(
            paper,
            profile,
            RankingConfig(w_embedding=0.0),
            QueriesConfig(),
            ["cs.CL"],
            embedding_score=0.9,
        )
        assert without["score_total"] == with_emb["score_total"]

    def test_embedding_score_none_ignored(self) -> None:
        paper = _make_paper()
        profile = _make_profile()
        result = score_paper(
            paper,
            profile,
            RankingConfig(w_embedding=1.5),
            QueriesConfig(),
            ["cs.CL"],
            embedding_score=None,
        )
        assert result["embedding_score"] is None
        assert result["score_total"] <= 1.0

    def test_rank_papers_with_repo_embedding(self) -> None:
        from unittest.mock import patch

        import numpy as np

        papers = [
            _make_paper(arxiv_id="2401.00001v1", title="Paper A"),
            _make_paper(arxiv_id="2401.00002v1", title="Paper B"),
        ]
        profile = _make_profile()
        repo_emb = np.array([1.0, 0.0, 0.0])

        with (
            patch("reporadar.embeddings.compute_paper_embedding") as mock_emb,
            patch("reporadar.embeddings.cosine_similarity") as mock_cos,
        ):
            mock_emb.side_effect = [np.array([0.9, 0.1, 0.0]), np.array([0.0, 0.0, 1.0])]
            mock_cos.side_effect = [0.99, 0.0]

            scores = rank_papers(
                papers,
                profile,
                RankingConfig(w_embedding=1.5),
                QueriesConfig(),
                ["cs.CL"],
                repo_embedding=repo_emb,
            )

        assert len(scores) == 2
        # Paper A should have embedding_score
        paper_a = next(s for s in scores if s["arxiv_id"] == "2401.00001v1")
        assert paper_a["embedding_score"] is not None


class TestPerCategoryWeights:
    def test_weighted_category_higher_score(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        # cs.CL weighted 2.0, cs.LG weighted 1.0
        score_weighted = score_category_match(
            paper, ["cs.CL", "cs.LG"], category_weights={"cs.CL": 2.0, "cs.LG": 1.0}
        )
        score_default = score_category_match(paper, ["cs.CL", "cs.LG"])
        # With cs.CL weighted 2x, matching just cs.CL should give 2/3 vs 1/2
        assert score_weighted > score_default

    def test_equal_weights_matches_default(self) -> None:
        paper = _make_paper(categories=["cs.CL"])
        score_weighted = score_category_match(
            paper, ["cs.CL", "cs.LG"], category_weights={"cs.CL": 1.0, "cs.LG": 1.0}
        )
        score_default = score_category_match(paper, ["cs.CL", "cs.LG"])
        assert score_weighted == pytest.approx(score_default)

    def test_unweighted_category_uses_default(self) -> None:
        paper = _make_paper(categories=["cs.AI"])
        # cs.AI not in weights dict → defaults to 1.0
        score = score_category_match(paper, ["cs.CL", "cs.AI"], category_weights={"cs.CL": 2.0})
        # cs.AI matches with weight 1.0, cs.CL doesn't match: 1.0 / (2.0 + 1.0) = 0.333
        assert score == pytest.approx(1.0 / 3.0, abs=0.01)
