"""Tests for reporadar.sources.openalex."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from reporadar.sources.openalex import (
    _extract_arxiv_id,
    _normalize_paper,
    collect_papers,
    reconstruct_abstract,
    search_papers,
)


def _mock_response(data: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_oa_work(**overrides) -> dict:
    base = {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.48550/arXiv.2401.12345",
        "title": "Test Paper",
        "display_name": "Test Paper",
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ],
        "abstract_inverted_index": {
            "A": [0],
            "test": [1],
            "abstract": [2],
            "about": [3],
            "machine": [4],
            "learning": [5],
        },
        "primary_topic": {"display_name": "Machine Learning"},
        "publication_date": "2026-01-15",
        "open_access": {"oa_url": "https://arxiv.org/pdf/2401.12345"},
        "ids": {"openalex": "https://openalex.org/W12345"},
    }
    base.update(overrides)
    return base


class TestReconstructAbstract:
    def test_basic_reconstruction(self) -> None:
        inv_index = {"Hello": [0], "world": [1], "of": [2], "ML": [3]}
        result = reconstruct_abstract(inv_index)
        assert result == "Hello world of ML"

    def test_repeated_words(self) -> None:
        inv_index = {"the": [0, 2], "cat": [1], "dog": [3]}
        result = reconstruct_abstract(inv_index)
        assert result == "the cat the dog"

    def test_empty_index(self) -> None:
        assert reconstruct_abstract(None) == ""
        assert reconstruct_abstract({}) == ""


class TestExtractArxivId:
    def test_from_doi(self) -> None:
        work = {"doi": "https://doi.org/10.48550/arXiv.2401.12345", "id": "x", "ids": {}}
        assert _extract_arxiv_id(work) == "2401.12345"

    def test_synthetic_id(self) -> None:
        work = {
            "doi": "",
            "id": "https://openalex.org/W99999",
            "ids": {"openalex": "https://openalex.org/W99999"},
        }
        result = _extract_arxiv_id(work)
        assert result == "oa:W99999"

    def test_no_id_at_all(self) -> None:
        work = {"doi": "", "id": "", "ids": {}}
        assert _extract_arxiv_id(work) == ""


class TestSearchPapers:
    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_returns_normalized_papers(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": [_make_oa_work()]})

        results = search_papers("machine learning")

        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2401.12345"
        assert results[0]["title"] == "Test Paper"
        assert "Alice Smith" in results[0]["authors"]

    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_query_url_construction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})

        search_papers("test query", email="user@example.com")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "mailto=user%40example.com" in req.full_url
        assert "search=test+query" in req.full_url

    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_abstract_reconstruction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": [_make_oa_work()]})

        results = search_papers("test")

        assert "test abstract" in results[0]["abstract"]


class TestPaperNormalization:
    def test_full_normalization(self) -> None:
        work = _make_oa_work()
        result = _normalize_paper(work)
        assert result is not None
        assert result["arxiv_id"] == "2401.12345"
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["Alice Smith", "Bob Jones"]
        assert "test abstract" in result["abstract"]
        assert result["published"].startswith("2026-01-15")

    def test_none_on_missing_title(self) -> None:
        work = _make_oa_work(title="", display_name="")
        result = _normalize_paper(work)
        assert result is None


class TestCollectPapers:
    @patch("reporadar.sources.openalex.time.sleep")
    @patch("reporadar.sources.openalex.search_papers")
    def test_dedup_across_queries(self, mock_search: MagicMock, mock_sleep: MagicMock) -> None:
        paper = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "authors": ["Alice"],
            "abstract": "abstract",
            "categories": [],
            "published": "2026-02-20T00:00:00+00:00",
            "updated": None,
            "url": "http://arxiv.org/abs/2401.12345",
            "pdf_url": None,
        }
        mock_search.return_value = [paper]

        results = collect_papers(["query1", "query2"], rate_limit=0.0)
        assert len(results) == 1

    @patch("reporadar.sources.openalex.time.sleep")
    @patch("reporadar.sources.openalex.search_papers")
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
            "published": "2026-02-20T00:00:00+00:00",
            "updated": None,
            "url": "",
            "pdf_url": None,
        }
        mock_search.return_value = [old_paper, new_paper]

        results = collect_papers(["q1"], lookback_days=30, rate_limit=0.0)

        arxiv_ids = [p["arxiv_id"] for p in results]
        assert "new" in arxiv_ids
        assert "old" not in arxiv_ids
