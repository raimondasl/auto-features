"""Tests for reporadar.paperswithcode."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from reporadar.paperswithcode import (
    fetch_enrichment,
    fetch_enrichments_batch,
    fetch_paper_info,
)


def _mock_response(data: object) -> MagicMock:
    """Create a mock urlopen response returning JSON data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestFetchPaperInfo:
    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_returns_first_result(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(
            {"results": [{"id": "paper-123", "title": "Test Paper"}]}
        )
        result = fetch_paper_info("2401.12345v1")
        assert result is not None
        assert result["id"] == "paper-123"

    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_correct_url_construction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})
        fetch_paper_info("2401.12345v1")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "arxiv_id=2401.12345" in req.full_url
        # Version suffix should be stripped from the arxiv_id parameter
        assert "arxiv_id=2401.12345v1" not in req.full_url

    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_returns_none_on_empty_results(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})
        result = fetch_paper_info("2401.99999v1")
        assert result is None

    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_returns_none_on_404(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
        result = fetch_paper_info("2401.99999v1")
        assert result is None


class TestFetchEnrichment:
    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_combines_paper_repos_datasets_tasks(self, mock_urlopen: MagicMock) -> None:
        paper_resp = _mock_response({"results": [{"id": "my-paper", "title": "Test"}]})
        repos_resp = _mock_response({"results": [{"url": "https://github.com/foo/bar"}]})
        datasets_resp = _mock_response({"results": [{"name": "ImageNet"}]})
        tasks_resp = _mock_response({"results": [{"name": "Image Classification"}]})
        mock_urlopen.side_effect = [paper_resp, repos_resp, datasets_resp, tasks_resp]

        result = fetch_enrichment("2401.12345v1")

        assert result is not None
        assert result["arxiv_id"] == "2401.12345v1"
        assert result["pwc_id"] == "my-paper"
        assert result["has_code"] is True
        assert result["code_urls"] == ["https://github.com/foo/bar"]
        assert result["datasets"] == ["ImageNet"]
        assert result["tasks"] == ["Image Classification"]

    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_no_code_repos(self, mock_urlopen: MagicMock) -> None:
        paper_resp = _mock_response({"results": [{"id": "my-paper"}]})
        repos_resp = _mock_response({"results": []})
        datasets_resp = _mock_response({"results": []})
        tasks_resp = _mock_response({"results": []})
        mock_urlopen.side_effect = [paper_resp, repos_resp, datasets_resp, tasks_resp]

        result = fetch_enrichment("2401.12345v1")

        assert result is not None
        assert result["has_code"] is False
        assert result["code_urls"] == []

    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_returns_none_when_paper_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})
        result = fetch_enrichment("2401.99999v1")
        assert result is None


class TestFetchEnrichmentsBatch:
    @patch("reporadar.paperswithcode.time.sleep")
    @patch("reporadar.paperswithcode.fetch_enrichment")
    def test_batch_processing(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        _e = {"code_urls": [], "datasets": [], "tasks": []}
        mock_fetch.side_effect = [
            {"arxiv_id": "a", "pwc_id": "pa", "has_code": True, **_e},
            {"arxiv_id": "b", "pwc_id": "pb", "has_code": False, **_e},
        ]

        result = fetch_enrichments_batch(["a", "b"], rate_limit=0.0)

        assert len(result) == 2
        assert "a" in result
        assert "b" in result

    @patch("reporadar.paperswithcode.time.sleep")
    @patch("reporadar.paperswithcode.fetch_enrichment")
    def test_partial_failure(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        _e = {"code_urls": [], "datasets": [], "tasks": []}
        mock_fetch.side_effect = [
            {"arxiv_id": "a", "pwc_id": "pa", "has_code": True, **_e},
            None,  # paper b not found
            {"arxiv_id": "c", "pwc_id": "pc", "has_code": False, **_e},
        ]

        result = fetch_enrichments_batch(["a", "b", "c"], rate_limit=0.0)

        assert len(result) == 2
        assert "a" in result
        assert "b" not in result
        assert "c" in result

    @patch("reporadar.paperswithcode.time.sleep")
    @patch("reporadar.paperswithcode.fetch_enrichment")
    def test_exception_handling(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        _e = {"code_urls": [], "datasets": [], "tasks": []}
        mock_fetch.side_effect = [
            Exception("network error"),
            {"arxiv_id": "b", "pwc_id": "pb", "has_code": False, **_e},
        ]

        result = fetch_enrichments_batch(["a", "b"], rate_limit=0.0)

        assert len(result) == 1
        assert "b" in result


class TestApiFailure:
    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("network error")
        result = fetch_paper_info("2401.12345v1")
        assert result is None

    @patch("reporadar.paperswithcode.time.sleep")
    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_rate_limit_retries(self, mock_urlopen: MagicMock, mock_sleep: MagicMock) -> None:
        import urllib.error

        # First: 429, then success
        success_resp = _mock_response({"results": [{"id": "paper-1"}]})
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
            success_resp,
        ]

        result = fetch_paper_info("2401.12345v1")

        assert result is not None
        assert result["id"] == "paper-1"
        mock_sleep.assert_called_once()

    @patch("reporadar.paperswithcode.time.sleep")
    @patch("reporadar.paperswithcode.urllib.request.urlopen")
    def test_server_error_retries(self, mock_urlopen: MagicMock, mock_sleep: MagicMock) -> None:
        import urllib.error

        success_resp = _mock_response({"results": [{"id": "paper-1"}]})
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 500, "Server Error", {}, None),
            success_resp,
        ]

        result = fetch_paper_info("2401.12345v1")

        assert result is not None
        mock_sleep.assert_called_once()
