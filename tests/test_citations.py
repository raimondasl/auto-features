"""Tests for reporadar.citations."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from reporadar.citations import fetch_citation_counts, normalize_citations


class TestFetchCitationCounts:
    @patch("reporadar.citations.urllib.request.urlopen")
    def test_basic_fetch(self, mock_urlopen: MagicMock) -> None:
        response_data = [
            {"paperId": "abc", "citationCount": 42},
            {"paperId": "def", "citationCount": 10},
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = fetch_citation_counts(["2401.00001v1", "2401.00002v1"])

        assert result == {"2401.00001v1": 42, "2401.00002v1": 10}
        mock_urlopen.assert_called_once()

    @patch("reporadar.citations.urllib.request.urlopen")
    def test_api_call_format(self, mock_urlopen: MagicMock) -> None:
        response_data = [{"paperId": "abc", "citationCount": 5}]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        fetch_citation_counts(["2401.12345v1"])

        # Verify the request was made with correct URL
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "semanticscholar.org" in req.full_url
        assert "citationCount" in req.full_url
        # Verify payload contains ARXIV: format
        payload = json.loads(req.data)
        assert "ARXIV:2401.12345" in payload["ids"]

    @patch("reporadar.citations.urllib.request.urlopen")
    def test_api_failure_returns_empty(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("network error")

        result = fetch_citation_counts(["2401.00001v1"])

        assert result == {}

    @patch("reporadar.citations.time.sleep")
    @patch("reporadar.citations.urllib.request.urlopen")
    def test_rate_limiting_handled(self, mock_urlopen: MagicMock, mock_sleep: MagicMock) -> None:
        import urllib.error

        # First call: 429 rate limited, second call: success
        error_resp = MagicMock()
        error_resp.code = 429
        error_resp.reason = "Too Many Requests"
        error_resp.read.return_value = b""

        response_data = [{"paperId": "abc", "citationCount": 7}]
        success_resp = MagicMock()
        success_resp.read.return_value = json.dumps(response_data).encode()
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None),
            success_resp,
        ]

        result = fetch_citation_counts(["2401.00001v1"])

        assert result == {"2401.00001v1": 7}
        mock_sleep.assert_called_once()

    def test_empty_list_returns_empty(self) -> None:
        result = fetch_citation_counts([])
        assert result == {}

    @patch("reporadar.citations.urllib.request.urlopen")
    def test_null_entries_skipped(self, mock_urlopen: MagicMock) -> None:
        response_data = [
            {"paperId": "abc", "citationCount": 42},
            None,  # Paper not found
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = fetch_citation_counts(["2401.00001v1", "2401.00002v1"])

        assert "2401.00001v1" in result
        assert "2401.00002v1" not in result


class TestNormalizeCitations:
    def test_basic_normalization(self) -> None:
        counts = {"a": 0, "b": 10, "c": 100}
        result = normalize_citations(counts)

        assert result["a"] == 0.0
        assert 0 < result["b"] < result["c"]
        assert result["c"] == pytest.approx(1.0)

    def test_log_formula(self) -> None:
        counts = {"a": 10, "b": 100}
        result = normalize_citations(counts)

        expected_a = math.log(1 + 10) / math.log(1 + 100)
        assert result["a"] == pytest.approx(expected_a)
        assert result["b"] == pytest.approx(1.0)

    def test_empty_returns_empty(self) -> None:
        assert normalize_citations({}) == {}

    def test_all_zeros(self) -> None:
        counts = {"a": 0, "b": 0}
        result = normalize_citations(counts)
        assert result["a"] == 0.0
        assert result["b"] == 0.0

    def test_single_entry(self) -> None:
        result = normalize_citations({"a": 50})
        assert result["a"] == pytest.approx(1.0)
