"""Tests for reporadar.citations."""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from reporadar.citations import fetch_citation_counts, fetch_references, normalize_citations


class TestFetchReferences:
    @patch("reporadar.citations.urllib.request.urlopen")
    def test_extracts_arxiv_references(self, mock_urlopen: MagicMock) -> None:
        response_data = [
            {
                "references": [
                    {"externalIds": {"ArXiv": "2401.00099", "DOI": "d"}},
                    {"externalIds": {"ArXiv": "2401.00042v2"}},  # version-stripped
                    {"externalIds": {"DOI": "no-arxiv"}},  # skipped
                    {"externalIds": None},  # skipped
                    None,  # skipped
                ]
            },
            None,  # this paper not found → absent from result
        ]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        out = fetch_references(["2402.11111v1", "2402.22222v1"])

        assert out == {"2402.11111v1": ["2401.00099", "2401.00042"]}
        assert "references.externalIds" in mock_urlopen.call_args[0][0].full_url

    def test_empty_input(self) -> None:
        assert fetch_references([]) == {}

    @patch("reporadar.citations.urllib.request.urlopen")
    def test_api_failure_returns_empty(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("network error")
        assert fetch_references(["2401.00001v1"]) == {}

    @patch("reporadar.citations._s2_batch_post")
    def test_chunks_small_enough_to_stay_under_the_nested_cap(self, mock_post: MagicMock) -> None:
        """Chunk size is bounded by nested items, not by the 500-id limit.

        The batch endpoint truncates nested results at 9,999 per request, filled greedily
        in id order, so a 500-id chunk asking for `references.externalIds` returns the
        first few papers' references and blanks the rest — HTTP 200, no error. Measured on
        300 real ids: 9,999 references with 85 of 300 papers blank, against 14,041 with 8
        blank at chunks of 50. This test previously asserted the 500 chunking, i.e. it
        encoded the bug.
        """
        from reporadar.citations import _S2_NESTED_CAP, _S2_REFERENCE_CHUNK

        mock_post.return_value = []
        ids = [f"2401.{i:05d}v1" for i in range(220)]
        fetch_references(ids)
        assert all(len(call.args[0]) <= _S2_REFERENCE_CHUNK for call in mock_post.call_args_list)
        # A chunk must not be able to reach the cap on any plausible reference density.
        assert _S2_REFERENCE_CHUNK * 100 < _S2_NESTED_CAP

    @patch("reporadar.citations._s2_batch_post")
    def test_a_chunk_pinned_at_the_cap_is_split_and_retried(self, mock_post: MagicMock) -> None:
        """Density varies by corpus, so a fixed chunk size is not enough on its own.

        A response whose nested total sits exactly at the cap is indistinguishable from a
        genuine result unless it is re-fetched smaller — and accepting it would silently
        drop the tail of the chunk.
        """
        from reporadar.citations import _S2_NESTED_CAP

        ref = {"externalIds": {"ArXiv": "1706.03762"}}
        calls: list[int] = []

        def fake_post(chunk, fields, *a, **k):  # type: ignore[no-untyped-def]
            calls.append(len(chunk))
            if len(chunk) > 1:
                # Simulate the cap: everything the API can fit, on the first paper only.
                return [{"references": [ref] * _S2_NESTED_CAP}] + [
                    {"references": []} for _ in chunk[1:]
                ]
            return [{"references": [ref]}]

        mock_post.side_effect = fake_post
        result = fetch_references([f"2401.{i:05d}v1" for i in range(8)])

        assert max(calls) > min(calls), "a capped chunk was accepted instead of being split"
        # Every paper resolves once the chunk is small enough, instead of only the first.
        assert len(result) == 8, f"only {len(result)} of 8 papers survived truncation"

    @patch("reporadar.citations._s2_batch_post")
    def test_an_ordinary_response_is_not_split(self, mock_post: MagicMock) -> None:
        # The split must be triggered by the cap, not by every request — otherwise it
        # multiplies traffic against an API that already rate-limits this project.
        ref = {"externalIds": {"ArXiv": "1706.03762"}}
        mock_post.return_value = [{"references": [ref]} for _ in range(10)]
        fetch_references([f"2401.{i:05d}v1" for i in range(10)])
        assert mock_post.call_count == 1

    @patch("reporadar.citations._s2_batch_post")
    def test_stats_distinguish_an_outage_from_a_genuine_negative(
        self, mock_post: MagicMock
    ) -> None:
        """An empty result means two very different things, and callers judge features on it.

        `seeded.py` reports "no signal" from an empty dict, so a rate-limited run reads as
        evidence about Feature 8 rather than as an outage. Its docstring already claimed to
        tell these apart; the information was being discarded inside `fetch_references`.
        """
        mock_post.return_value = None  # every request fails
        stats: dict[str, int] = {}
        out = fetch_references([f"2401.{i:05d}v1" for i in range(20)], stats=stats)
        assert out == {}
        assert stats["requests"] > 0
        assert stats["failed"] == stats["requests"], "an outage must be visible in the stats"

    @patch("reporadar.citations._s2_batch_post")
    def test_stats_show_no_failures_on_a_genuine_empty_result(self, mock_post: MagicMock) -> None:
        # Papers that really cite nothing arXiv-indexed must NOT look like an outage.
        mock_post.return_value = [{"references": []} for _ in range(20)]
        stats: dict[str, int] = {}
        assert fetch_references([f"2401.{i:05d}v1" for i in range(20)], stats=stats) == {}
        assert stats["failed"] == 0


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
