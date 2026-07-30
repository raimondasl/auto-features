"""Tests for reporadar.sources.hf_papers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from reporadar.sources.hf_papers import (
    _base_arxiv_id,
    _code_urls_from_paper,
    _repo_ids,
    fetch_enrichment,
    fetch_enrichments_batch,
    normalize_upvotes,
)


def _mock_response(data: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestHelpers:
    def test_base_arxiv_id_strips_version(self) -> None:
        assert _base_arxiv_id("2401.12345v2") == "2401.12345"
        assert _base_arxiv_id("2401.12345") == "2401.12345"

    def test_repo_ids_extracts_and_caps(self) -> None:
        items = [{"id": f"org/model{i}"} for i in range(20)]
        ids = _repo_ids(items)
        assert len(ids) == 10  # MAX_ITEMS
        assert ids[0] == "org/model0"

    def test_repo_ids_handles_modelid_and_junk(self) -> None:
        assert _repo_ids([{"modelId": "a/b"}, {"nope": 1}, "junk"]) == ["a/b"]
        assert _repo_ids(None) == []

    def test_code_urls_from_github_repo(self) -> None:
        assert _code_urls_from_paper({"githubRepo": "google/bert"}) == [
            "https://github.com/google/bert"
        ]

    def test_code_urls_full_url_passthrough(self) -> None:
        assert _code_urls_from_paper({"githubRepo": "https://github.com/x/y"}) == [
            "https://github.com/x/y"
        ]

    def test_code_urls_empty_when_absent(self) -> None:
        assert _code_urls_from_paper({}) == []


class TestFetchEnrichment:
    @patch("reporadar.sources.hf_papers.urllib.request.urlopen")
    def test_full_enrichment(self, mock_urlopen: MagicMock) -> None:
        # Order of calls: paper, models, datasets
        mock_urlopen.side_effect = [
            _mock_response({"id": "2401.12345", "upvotes": 42, "githubRepo": "org/repo"}),
            _mock_response([{"id": "org/model-a"}, {"id": "org/model-b"}]),
            _mock_response([{"id": "org/dataset-a"}]),
        ]

        result = fetch_enrichment("2401.12345")

        assert result is not None
        assert result["arxiv_id"] == "2401.12345"
        assert result["hf_id"] == "2401.12345"
        assert result["upvotes"] == 42
        assert result["has_code"] is True
        assert result["code_urls"] == ["https://github.com/org/repo"]
        assert result["models"] == ["org/model-a", "org/model-b"]
        assert result["datasets"] == ["org/dataset-a"]
        assert result["tasks"] == []

    @patch("reporadar.sources.hf_papers.urllib.request.urlopen")
    def test_models_only_no_paper_page(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        # Paper page 404s, but models exist -> still enriched.
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("u", 404, "nf", {}, None),  # type: ignore[arg-type]
            _mock_response([{"id": "org/model-a"}]),
            _mock_response([]),
        ]

        result = fetch_enrichment("2401.99999")

        assert result is not None
        assert result["upvotes"] == 0
        assert result["has_code"] is False
        assert result["models"] == ["org/model-a"]

    @patch("reporadar.sources.hf_papers.urllib.request.urlopen")
    def test_returns_none_when_nothing_found(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = [
            urllib.error.HTTPError("u", 404, "nf", {}, None),  # type: ignore[arg-type]
            _mock_response([]),
            _mock_response([]),
        ]

        assert fetch_enrichment("0000.00000") is None


class TestFetchEnrichmentsBatch:
    @patch("reporadar.sources.hf_papers.time.sleep")
    @patch("reporadar.sources.hf_papers.fetch_enrichment")
    def test_batch_collects_and_skips_none(
        self, mock_fetch: MagicMock, mock_sleep: MagicMock
    ) -> None:
        mock_fetch.side_effect = [
            {"arxiv_id": "a", "has_code": True, "models": [], "datasets": [], "upvotes": 1},
            None,
        ]

        results = fetch_enrichments_batch(["a", "b"], rate_limit=0.0)

        assert set(results.keys()) == {"a"}

    @patch("reporadar.sources.hf_papers.time.sleep")
    @patch("reporadar.sources.hf_papers.fetch_enrichment")
    def test_batch_tolerates_exceptions(self, mock_fetch: MagicMock, mock_sleep: MagicMock) -> None:
        mock_fetch.side_effect = [
            RuntimeError("boom"),
            {"arxiv_id": "b", "has_code": False, "models": [], "datasets": [], "upvotes": 0},
        ]

        results = fetch_enrichments_batch(["a", "b"], rate_limit=0.0)

        assert set(results.keys()) == {"b"}


class TestNormalizeUpvotes:
    def test_scales_relative_to_the_run_maximum(self) -> None:
        scores = normalize_upvotes({"a": 100, "b": 10, "c": 1})
        assert scores["a"] == 1.0  # the run's leader anchors the scale
        assert 0.0 < scores["c"] < scores["b"] < scores["a"]

    def test_log_scale_beats_linear_for_the_tail(self) -> None:
        # Linear would put a 10-upvote paper at 0.1; log keeps the tail legible.
        scores = normalize_upvotes({"leader": 1000, "tail": 10})
        assert scores["tail"] > 0.3

    def test_zero_upvotes_are_omitted_not_scored_zero(self) -> None:
        # A 0 usually means "HF has no page for this paper" — an absent signal.
        # Scoring it as 0.0 would let w_community silently penalize the paper.
        scores = normalize_upvotes({"a": 5, "b": 0})
        assert set(scores) == {"a"}

    def test_empty_and_all_zero_inputs(self) -> None:
        assert normalize_upvotes({}) == {}
        assert normalize_upvotes({"a": 0, "b": 0}) == {}

    def test_single_paper_anchors_itself(self) -> None:
        assert normalize_upvotes({"a": 7}) == {"a": 1.0}
