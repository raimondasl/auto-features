"""Tests for reporadar.collector."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import arxiv

from reporadar.collector import (
    _category_filter,
    _result_to_paper,
    build_queries,
    collect_papers,
)
from reporadar.config import ArxivConfig, QueriesConfig
from reporadar.profiler import RepoProfile


def _make_profile(**overrides) -> RepoProfile:
    defaults = {
        "keywords": [("transformers", 0.8), ("retrieval", 0.6), ("generation", 0.5)],
        "anchors": ["torch", "transformers"],
        "domains": ["deep learning", "NLP"],
    }
    defaults.update(overrides)
    return RepoProfile(**defaults)


def _make_arxiv_result(
    entry_id: str = "http://arxiv.org/abs/2401.12345v1",
    title: str = "Test Paper",
    summary: str = "A test abstract.",
    authors: list[str] | None = None,
    categories: list[str] | None = None,
    published: datetime | None = None,
) -> arxiv.Result:
    """Create a mock-like arxiv.Result for testing."""
    if authors is None:
        authors = ["Alice Smith"]
    if categories is None:
        categories = ["cs.CL"]
    if published is None:
        published = datetime.now(timezone.utc)

    result = arxiv.Result(
        entry_id=entry_id,
        title=title,
        summary=summary,
        published=published,
        updated=published,
        categories=categories,
    )
    # Set authors via Author objects
    result.authors = [arxiv.Result.Author(name) for name in authors]
    return result


class TestCategoryFilter:
    def test_single_category(self) -> None:
        assert _category_filter(["cs.LG"]) == "cat:cs.LG"

    def test_multiple_categories(self) -> None:
        result = _category_filter(["cs.LG", "cs.CL"])
        assert result == "cat:cs.LG OR cat:cs.CL"

    def test_empty(self) -> None:
        assert _category_filter([]) == ""


class TestBuildQueries:
    def test_seed_queries_included(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig(seed=["retrieval augmented generation"])
        arxiv_cfg = ArxivConfig(categories=["cs.CL"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert any('"retrieval augmented generation"' in q for q in queries)

    def test_auto_queries_from_keywords(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.LG"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert len(queries) > 0
        assert any("transformers" in q for q in queries)

    def test_category_filter_applied(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.LG"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        for q in queries:
            assert "cat:cs.LG" in q

    def test_no_categories(self) -> None:
        profile = _make_profile()
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=[])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert len(queries) > 0
        for q in queries:
            assert "cat:" not in q

    def test_empty_profile_with_seeds(self) -> None:
        profile = _make_profile(keywords=[], anchors=[], domains=[])
        queries_cfg = QueriesConfig(seed=["test query"])
        arxiv_cfg = ArxivConfig(categories=["cs.AI"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        assert len(queries) == 1
        assert '"test query"' in queries[0]

    def test_empty_everything_fallback(self) -> None:
        profile = _make_profile(keywords=[], anchors=[], domains=[])
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.LG"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        # Should fall back to category-only query
        assert len(queries) == 1
        assert "cat:cs.LG" in queries[0]

    def test_max_auto_queries(self) -> None:
        keywords = [(f"term{i}", 0.9 - i * 0.1) for i in range(10)]
        profile = _make_profile(keywords=keywords)
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=[])

        queries = build_queries(profile, queries_cfg, arxiv_cfg, max_auto_queries=3)

        assert len(queries) == 3


class TestResultToPaper:
    def test_converts_fields(self) -> None:
        result = _make_arxiv_result(
            entry_id="http://arxiv.org/abs/2401.99999v1",
            title="My Paper",
            summary="Abstract text.",
            authors=["Alice", "Bob"],
            categories=["cs.CL", "cs.AI"],
        )

        paper = _result_to_paper(result)

        assert paper["arxiv_id"] == "2401.99999v1"
        assert paper["title"] == "My Paper"
        assert paper["abstract"] == "Abstract text."
        assert paper["authors"] == ["Alice", "Bob"]
        assert paper["categories"] == ["cs.CL", "cs.AI"]
        assert "url" in paper
        assert paper["published"] is not None


class TestCollectPapers:
    @patch("reporadar.collector.arxiv.Client")
    def test_collects_and_deduplicates(self, MockClient: MagicMock) -> None:
        now = datetime.now(timezone.utc)
        results_q1 = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", "Paper A", published=now),
            _make_arxiv_result("http://arxiv.org/abs/2401.00002v1", "Paper B", published=now),
        ]
        results_q2 = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00002v1", "Paper B", published=now),  # duplicate
            _make_arxiv_result("http://arxiv.org/abs/2401.00003v1", "Paper C", published=now),
        ]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = [iter(results_q1), iter(results_q2)]

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=30)
        papers = collect_papers(["q1", "q2"], cfg)

        assert len(papers) == 3
        ids = {p["arxiv_id"] for p in papers}
        assert ids == {"2401.00001v1", "2401.00002v1", "2401.00003v1"}

    @patch("reporadar.collector.arxiv.Client")
    def test_filters_old_papers(self, MockClient: MagicMock) -> None:
        now = datetime.now(timezone.utc)
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)

        results = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", "New Paper", published=now),
            _make_arxiv_result("http://arxiv.org/abs/2001.00001v1", "Old Paper", published=old),
        ]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter(results)

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=14)
        papers = collect_papers(["q1"], cfg)

        assert len(papers) == 1
        assert papers[0]["title"] == "New Paper"

    @patch("reporadar.collector.arxiv.Client")
    def test_matched_query_recorded(self, MockClient: MagicMock) -> None:
        now = datetime.now(timezone.utc)
        results = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", published=now),
        ]

        mock_client = MockClient.return_value
        mock_client.results.return_value = iter(results)

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=30)
        papers = collect_papers(["all:transformers"], cfg)

        assert papers[0]["matched_query"] == "all:transformers"

    @patch("reporadar.collector.arxiv.Client")
    def test_empty_results(self, MockClient: MagicMock) -> None:
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=14)
        papers = collect_papers(["q1"], cfg)

        assert papers == []
