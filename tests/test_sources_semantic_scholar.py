"""Tests for reporadar.sources.semantic_scholar."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from reporadar.sources.semantic_scholar import (
    _normalize_paper,
    collect_papers,
    search_papers,
)


def _mock_response(data: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_ss_paper(**overrides) -> dict:
    base = {
        "paperId": "abc123",
        "externalIds": {"ArXiv": "2401.12345"},
        "title": "Test Paper",
        "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
        "abstract": "A test abstract about machine learning.",
        "publicationTypes": ["JournalArticle"],
        "year": 2024,
        "url": "https://www.semanticscholar.org/paper/abc123",
        "citationCount": 42,
    }
    base.update(overrides)
    return base


class TestSearchPapers:
    @patch("reporadar.sources.semantic_scholar.urllib.request.urlopen")
    def test_returns_normalized_papers(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"data": [_make_ss_paper()]})

        results = search_papers("machine learning")

        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2401.12345"
        assert results[0]["title"] == "Test Paper"
        assert results[0]["authors"] == ["Alice Smith", "Bob Jones"]

    @patch("reporadar.sources.semantic_scholar.urllib.request.urlopen")
    def test_query_url_construction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"data": []})

        search_papers("retrieval augmented generation", limit=10)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "retrieval%20augmented%20generation" in req.full_url
        assert "limit=10" in req.full_url

    @patch("reporadar.sources.semantic_scholar.urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"data": []})
        results = search_papers("nonexistent query")
        assert results == []


class TestPaperNormalization:
    def test_arxiv_id_from_external_ids(self) -> None:
        paper = _make_ss_paper()
        result = _normalize_paper(paper)
        assert result is not None
        assert result["arxiv_id"] == "2401.12345"

    def test_synthetic_id_without_arxiv(self) -> None:
        paper = _make_ss_paper(externalIds={})
        result = _normalize_paper(paper)
        assert result is not None
        assert result["arxiv_id"] == "ss:abc123"

    def test_authors_extracted(self) -> None:
        paper = _make_ss_paper()
        result = _normalize_paper(paper)
        assert result is not None
        assert result["authors"] == ["Alice Smith", "Bob Jones"]

    def test_year_to_published_date(self) -> None:
        paper = _make_ss_paper(year=2024)
        result = _normalize_paper(paper)
        assert result is not None
        assert result["published"].startswith("2024-")

    def test_none_on_missing_title(self) -> None:
        paper = _make_ss_paper(title="")
        result = _normalize_paper(paper)
        assert result is None

    def test_none_on_missing_paper_id(self) -> None:
        paper = _make_ss_paper(paperId="", externalIds={})
        result = _normalize_paper(paper)
        assert result is None


class TestCollectPapers:
    @patch("reporadar.sources.semantic_scholar.time.sleep")
    @patch("reporadar.sources.semantic_scholar.search_papers")
    def test_dedup_across_queries(self, mock_search: MagicMock, mock_sleep: MagicMock) -> None:
        paper = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "authors": ["Alice"],
            "abstract": "abstract",
            "categories": [],
            "published": "2026-01-20T00:00:00+00:00",
            "updated": None,
            "url": "http://arxiv.org/abs/2401.12345",
            "pdf_url": None,
        }
        mock_search.return_value = [paper]

        results = collect_papers(["query1", "query2"], rate_limit=0.0)

        assert len(results) == 1  # deduped

    @patch("reporadar.sources.semantic_scholar.time.sleep")
    @patch("reporadar.sources.semantic_scholar.search_papers")
    def test_date_filtering(self, mock_search: MagicMock, mock_sleep: MagicMock) -> None:
        old_paper = {
            "arxiv_id": "old",
            "title": "Old",
            "authors": [],
            "abstract": "",
            "categories": [],
            "published": "2020-01-01T00:00:00+00:00",
            "updated": None,
            "url": "",
            "pdf_url": None,
        }
        new_paper = {
            "arxiv_id": "new",
            "title": "New",
            "authors": [],
            "abstract": "",
            "categories": [],
            "published": "2026-01-01T00:00:00+00:00",
            "updated": None,
            "url": "",
            "pdf_url": None,
        }
        mock_search.return_value = [old_paper, new_paper]

        results = collect_papers(["q1"], lookback_days=365, rate_limit=0.0)

        arxiv_ids = [p["arxiv_id"] for p in results]
        assert "new" in arxiv_ids
        assert "old" not in arxiv_ids


class TestApiFailure:
    @patch("reporadar.sources.semantic_scholar.urllib.request.urlopen")
    def test_network_error_returns_empty(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("network error")
        results = search_papers("test")
        assert results == []

    @patch("reporadar.sources.semantic_scholar.time.sleep")
    @patch("reporadar.sources.semantic_scholar.urllib.request.urlopen")
    def test_rate_limit_retries(self, mock_urlopen: MagicMock, mock_sleep: MagicMock) -> None:
        import urllib.error

        success_resp = _mock_response({"data": [_make_ss_paper()]})
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
            success_resp,
        ]

        results = search_papers("test")
        assert len(results) == 1
        mock_sleep.assert_called_once()
