"""Tests for reporadar.ranker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reporadar.config import QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile
from reporadar.ranker import (
    compute_exclude_penalty,
    rank_papers,
    score_category_match,
    score_keyword_overlap,
    score_paper,
    score_recency,
)


def _make_paper(**overrides) -> dict:
    now = datetime.now(timezone.utc)
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
        profile = _make_profile(
            keywords=[("retrieval", 0.5), ("transformers", 0.5)]
        )
        score = score_keyword_overlap(paper, profile)
        assert score == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        paper = _make_paper()
        profile = _make_profile(
            keywords=[("retrieval", 0.5), ("quantum", 0.5)]
        )
        score = score_keyword_overlap(paper, profile)
        assert 0.4 < score < 0.6

    def test_no_overlap(self) -> None:
        paper = _make_paper(
            title="Quantum Computing Advances",
            abstract="A new quantum error correction code.",
        )
        profile = _make_profile(
            keywords=[("retrieval", 0.5), ("transformers", 0.5)]
        )
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
        now = datetime.now(timezone.utc)
        paper = _make_paper(published=now.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score > 0.95

    def test_old_paper_is_0(self) -> None:
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        paper = _make_paper(published=old.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert score == 0.0

    def test_midpoint(self) -> None:
        mid = datetime.now(timezone.utc) - timedelta(days=7)
        paper = _make_paper(published=mid.isoformat())
        score = score_recency(paper, lookback_days=14)
        assert 0.4 < score < 0.6

    def test_decays_linearly(self) -> None:
        now = datetime.now(timezone.utc)
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
            paper, profile,
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
            paper, profile,
            RankingConfig(w_keyword=10.0, w_category=0.0, w_recency=0.0),
            QueriesConfig(),
            ["cs.CL"],
        )
        # Heavy recency weight
        rec_heavy = score_paper(
            paper, profile,
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
            paper, profile, RankingConfig(), QueriesConfig(), ["cs.CL"],
        )
        with_exclude = score_paper(
            paper, profile, RankingConfig(), QueriesConfig(exclude=["survey"]), ["cs.CL"],
        )

        assert with_exclude["score_total"] < without_exclude["score_total"]


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
            papers, profile, RankingConfig(), QueriesConfig(), ["cs.CL"],
        )

        assert len(scores) == 2
        assert scores[0]["score_total"] >= scores[1]["score_total"]
        # The RAG paper should rank higher
        assert scores[0]["arxiv_id"] == "2401.00002v1"

    def test_empty_papers(self) -> None:
        profile = _make_profile()
        scores = rank_papers(
            [], profile, RankingConfig(), QueriesConfig(), ["cs.CL"],
        )
        assert scores == []
