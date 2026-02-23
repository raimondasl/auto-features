"""Tests for reporadar.collector."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import arxiv
import pytest

from reporadar.collector import (
    CollectionError,
    _category_filter,
    _generate_bigram_queries,
    _query_with_retry,
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
        published = datetime.now(UTC)

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

        # Count only single-keyword queries (exclude bigram phrase queries)
        keyword_queries = [q for q in queries if '"' not in q]
        assert len(keyword_queries) == 3


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
        now = datetime.now(UTC)
        results_q1 = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", "Paper A", published=now),
            _make_arxiv_result("http://arxiv.org/abs/2401.00002v1", "Paper B", published=now),
        ]
        results_q2 = [
            _make_arxiv_result(  # duplicate
                "http://arxiv.org/abs/2401.00002v1",
                "Paper B",
                published=now,
            ),
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
        now = datetime.now(UTC)
        old = datetime(2020, 1, 1, tzinfo=UTC)

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
        now = datetime.now(UTC)
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

    @patch("reporadar.collector.arxiv.Client")
    def test_on_query_start_callback(self, MockClient: MagicMock) -> None:
        now = datetime.now(UTC)
        results = [
            _make_arxiv_result("http://arxiv.org/abs/2401.00001v1", published=now),
        ]

        mock_client = MockClient.return_value
        mock_client.results.side_effect = [iter(results), iter([])]

        calls: list[tuple[int, int, str]] = []

        def callback(idx: int, total: int, query: str) -> None:
            calls.append((idx, total, query))

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=30)
        collect_papers(["q1", "q2"], cfg, on_query_start=callback)

        assert len(calls) == 2
        assert calls[0] == (0, 2, "q1")
        assert calls[1] == (1, 2, "q2")

    @patch("reporadar.collector.arxiv.Client")
    def test_no_callback_by_default(self, MockClient: MagicMock) -> None:
        mock_client = MockClient.return_value
        mock_client.results.return_value = iter([])

        cfg = ArxivConfig(max_results_per_query=50, lookback_days=14)
        # Should not raise when callback is None (default)
        papers = collect_papers(["q1"], cfg)
        assert papers == []


class TestQueryWithRetry:
    @patch("reporadar.collector.time.sleep")
    def test_succeeds_after_transient_failure(self, mock_sleep: MagicMock) -> None:
        now = datetime.now(UTC)
        good_result = _make_arxiv_result(
            "http://arxiv.org/abs/2401.00001v1",
            published=now,
        )

        mock_client = MagicMock()
        mock_client.results.side_effect = [
            ConnectionError("network down"),
            [good_result],
        ]

        search = MagicMock()
        results = _query_with_retry(mock_client, search, max_retries=3, base_delay=1.0)

        assert len(results) == 1
        assert mock_sleep.call_count == 1
        # First retry delay should be base_delay * 2^0 = 1.0
        mock_sleep.assert_called_with(1.0)

    @patch("reporadar.collector.time.sleep")
    def test_exhausted_raises_collection_error(self, mock_sleep: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.results.side_effect = ConnectionError("always fails")

        search = MagicMock()
        with pytest.raises(CollectionError, match="3 attempts"):
            _query_with_retry(mock_client, search, max_retries=3, base_delay=1.0)

        assert mock_sleep.call_count == 2  # retries = max_retries - 1

    @patch("reporadar.collector.time.sleep")
    def test_backoff_delay_doubles(self, mock_sleep: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.results.side_effect = [
            TimeoutError("timeout"),
            OSError("network"),
            ConnectionError("fail"),
        ]

        search = MagicMock()
        with pytest.raises(CollectionError):
            _query_with_retry(mock_client, search, max_retries=3, base_delay=2.0)

        # Should have slept twice with exponential backoff
        assert mock_sleep.call_count == 2
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls[0] == 2.0  # base_delay * 2^0
        assert calls[1] == 4.0  # base_delay * 2^1


class TestBigramQueries:
    def test_bigrams_generated_from_top_keywords(self) -> None:
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        bigrams = _generate_bigram_queries(profile)
        assert len(bigrams) >= 1
        assert '"retrieval augmented"' in bigrams

    def test_bigrams_quoted_in_query(self) -> None:
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        bigrams = _generate_bigram_queries(profile)
        for b in bigrams:
            assert b.startswith('"') and b.endswith('"')

    def test_no_bigrams_from_single_keyword(self) -> None:
        profile = _make_profile(keywords=[("retrieval", 0.9)])
        bigrams = _generate_bigram_queries(profile)
        assert bigrams == []

    def test_bigrams_added_to_build_queries(self) -> None:
        profile = _make_profile(
            keywords=[("retrieval", 0.9), ("augmented", 0.8), ("generation", 0.7)]
        )
        queries_cfg = QueriesConfig()
        arxiv_cfg = ArxivConfig(categories=["cs.CL"])

        queries = build_queries(profile, queries_cfg, arxiv_cfg)

        # Should have bigram queries (quoted phrases) in the query list
        has_bigram = any('"retrieval augmented"' in q for q in queries)
        assert has_bigram

    def test_short_words_filtered(self) -> None:
        profile = _make_profile(keywords=[("an", 0.9), ("to", 0.8), ("transformers", 0.7)])
        bigrams = _generate_bigram_queries(profile)
        # "an to" should be filtered (both < 4 chars)
        assert '"an to"' not in bigrams

    def test_max_bigrams_respected(self) -> None:
        keywords = [(f"word{i:02d}", 0.9 - i * 0.05) for i in range(10)]
        profile = _make_profile(keywords=keywords)
        bigrams = _generate_bigram_queries(profile, max_bigrams=2)
        assert len(bigrams) <= 2
