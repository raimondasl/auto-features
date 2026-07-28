"""Tests for reporadar.sources.biorxiv (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reporadar.sources.biorxiv import collect_papers, fetch_window


def _item(doi: str, title: str = "T", abstract: str = "abs") -> dict:
    return {
        "doi": doi,
        "title": title,
        "authors": "Smith, J.; Jones, A.",
        "date": "2024-06-01",
        "category": "neuroscience",
        "abstract": abstract,
    }


def _paper(doi: str, title: str, abstract: str) -> dict:
    return {
        "arxiv_id": f"biorxiv:{doi}",
        "title": title,
        "authors": ["Smith, J."],
        "abstract": abstract,
        "categories": ["neuroscience"],
        "published": "2024-06-01T00:00:00+00:00",
        "updated": None,
        "url": f"https://www.biorxiv.org/content/{doi}",
        "pdf_url": None,
    }


class TestFetchWindow:
    @patch("reporadar.sources.biorxiv._request_json")
    def test_paginates_and_stops_at_total(self, mock_req: MagicMock) -> None:
        mock_req.side_effect = [
            {"messages": [{"total": "3"}], "collection": [_item("d1"), _item("d2")]},
            {"messages": [{"total": "3"}], "collection": [_item("d3")]},
        ]
        out = fetch_window(lookback_days=2)
        assert [p["arxiv_id"] for p in out] == ["biorxiv:d1", "biorxiv:d2", "biorxiv:d3"]
        assert mock_req.call_count == 2  # stopped once cursor (3) >= total (3)
        assert out[0]["authors"] == ["Smith, J.", "Jones, A."]

    @patch("reporadar.sources.biorxiv._request_json")
    def test_failure_returns_empty(self, mock_req: MagicMock) -> None:
        mock_req.return_value = None
        assert fetch_window(lookback_days=2) == []

    @patch("reporadar.sources.biorxiv._request_json")
    def test_stops_on_empty_collection(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"messages": [{"total": "99"}], "collection": []}
        assert fetch_window(lookback_days=2) == []
        assert mock_req.call_count == 1


class TestCollectPapers:
    @patch("reporadar.sources.biorxiv.fetch_window")
    def test_filters_by_query_terms(self, mock_fw: MagicMock) -> None:
        mock_fw.return_value = [
            _paper("d1", "CRISPR gene editing", "a method"),
            _paper("d2", "Totally unrelated", "nothing here"),
        ]
        out = collect_papers(["CRISPR"], lookback_days=2)
        assert [p["arxiv_id"] for p in out] == ["biorxiv:d1"]

    @patch("reporadar.sources.biorxiv.fetch_window")
    def test_empty_queries_returns_empty(self, mock_fw: MagicMock) -> None:
        assert collect_papers([], lookback_days=2) == []
        mock_fw.assert_not_called()  # no fetch when there are no query terms
