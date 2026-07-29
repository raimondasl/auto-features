"""Tests for reporadar.sources.s2_recommendations (mocked HTTP)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from reporadar.sources.s2_recommendations import _seed_ids, fetch_recommendations


def _resp(*papers: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"recommendedPapers": list(papers)}).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestSeedIds:
    def test_keeps_real_arxiv_ids_and_strips_versions(self) -> None:
        assert _seed_ids(["2401.12345v2", "cs.LG/0501001"], 10) == [
            "ARXIV:2401.12345",
            "ARXIV:cs.LG/0501001",
        ]

    def test_keeps_hyphenated_legacy_subject_classes(self) -> None:
        # e.g. cond-mat.stat-mech/0509001 is a real legacy arXiv id.
        assert _seed_ids(["cond-mat.stat-mech/0509001v1"], 10) == [
            "ARXIV:cond-mat.stat-mech/0509001"
        ]

    def test_version_strip_does_not_split_archive_containing_v(self) -> None:
        # solv-int is a real legacy archive; splitting on the first "v" would
        # produce the seed "ARXIV:sol", and one unresolvable positive seed makes
        # the whole call 400 — killing every recommendation for the run.
        assert _seed_ids(["solv-int/9502001v1"], 10) == ["ARXIV:solv-int/9502001"]

    def test_bare_s2_paper_id_seeds(self) -> None:
        # "ss:" wraps an S2 paperId, which this endpoint accepts directly — so
        # papers found *via* recommendations can seed later runs.
        sha = "4e53c699ae1b096b5b33e118371cd4c330ebc95e"
        assert _seed_ids([f"ss:{sha}"], 10) == [sha]

    def test_drops_unresolvable_synthetic_ids(self) -> None:
        # These would make the API return HTTP 400 ("No positive paper examples").
        assert _seed_ids(["dblp:conf/x/y", "biorxiv:10.1101/x", "ss:not-a-sha", "oa:W1"], 10) == []

    def test_dedups_and_caps(self) -> None:
        assert _seed_ids(["2401.00001v1", "2401.00001v2"], 10) == ["ARXIV:2401.00001"]
        assert len(_seed_ids([f"2401.{i:05d}" for i in range(20)], 5)) == 5


class TestFetchRecommendations:
    def test_no_positive_seeds_skips_the_call(self) -> None:
        with patch("reporadar.sources.s2_recommendations.urllib.request.urlopen") as mock_open:
            assert fetch_recommendations([], ["2401.1"]) == []
            assert fetch_recommendations(["dblp:only/synthetic"], []) == []
            mock_open.assert_not_called()  # negative-only / unusable seeds would 400

    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_normalizes_arxiv_and_non_arxiv(self, mock_open: MagicMock) -> None:
        mock_open.return_value = _resp(
            {
                "paperId": "p1",
                "title": "An arXiv Paper",
                "abstract": "abs",
                "externalIds": {"ArXiv": "2402.00001"},
                "authors": [{"name": "Alice"}],
                "year": 2024,
            },
            {
                "paperId": "p2",
                "title": "A Journal Paper",
                "externalIds": {"DOI": "10.1/x"},  # no ArXiv → synthetic id
                "authors": [],
                "openAccessPdf": {"url": "https://oa/pdf"},
            },
            {"paperId": "p3", "title": ""},  # no title → dropped
        )
        out = fetch_recommendations(["2106.09685"], ["1706.03762"])

        assert [p["arxiv_id"] for p in out] == ["2402.00001", "ss:p2"]
        assert all(p["matched_query"] == "recommendation" for p in out)
        assert out[0]["url"] == "http://arxiv.org/abs/2402.00001"
        assert out[0]["authors"] == ["Alice"]
        assert out[1]["pdf_url"] == "https://oa/pdf"

    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_uses_publication_date_over_year(self, mock_open: MagicMock) -> None:
        # A year-only date would look ~6 months stale to the recency scorer and
        # mute genuinely recent recommendations, so the exact date wins.
        mock_open.return_value = _resp(
            {
                "paperId": "p1",
                "title": "Recent",
                "externalIds": {"ArXiv": "2402.1"},
                "year": 2026,
                "publicationDate": "2026-07-20",
            },
            {
                "paperId": "p2",
                "title": "YearOnly",
                "externalIds": {"ArXiv": "2402.2"},
                "year": 2026,
            },
        )
        out = fetch_recommendations(["2106.09685"])
        assert out[0]["published"].startswith("2026-07-20")
        assert out[1]["published"].startswith("2026-01-01")  # year fallback

    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_sends_positive_and_negative_seeds(self, mock_open: MagicMock) -> None:
        mock_open.return_value = _resp()
        fetch_recommendations(["2106.09685v1"], ["1706.03762"], limit=5)
        req = mock_open.call_args[0][0]
        payload = json.loads(req.data)
        assert payload["positivePaperIds"] == ["ARXIV:2106.09685"]
        assert payload["negativePaperIds"] == ["ARXIV:1706.03762"]
        assert "limit=5" in req.full_url

    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_http_400_returns_none_not_empty(self, mock_open: MagicMock) -> None:
        # None = "request failed" (so the CLI can say *unavailable*); [] would
        # mean "the API genuinely had nothing for you".
        import urllib.error

        mock_open.side_effect = urllib.error.HTTPError("u", 400, "Bad", {}, None)  # type: ignore[arg-type]
        assert fetch_recommendations(["2106.09685"]) is None
        assert mock_open.call_count == 1  # a 400 is not retried

    @patch("reporadar.sources.s2_recommendations.time.sleep")
    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_retries_on_rate_limit_then_succeeds(
        self, mock_open: MagicMock, _sleep: MagicMock
    ) -> None:
        import urllib.error

        mock_open.side_effect = [
            urllib.error.HTTPError("u", 429, "Too Many", {}, None),  # type: ignore[arg-type]
            _resp({"paperId": "p1", "title": "T", "externalIds": {"ArXiv": "2402.1"}}),
        ]
        out = fetch_recommendations(["2106.09685"])
        assert out is not None and len(out) == 1
        assert mock_open.call_count == 2

    @patch("reporadar.sources.s2_recommendations.time.sleep")
    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_network_error_exhausts_retries_and_returns_none(
        self, mock_open: MagicMock, _sleep: MagicMock
    ) -> None:
        import urllib.error

        mock_open.side_effect = urllib.error.URLError("down")
        assert fetch_recommendations(["2106.09685"]) is None
        assert mock_open.call_count == 3

    @patch("reporadar.sources.s2_recommendations.urllib.request.urlopen")
    def test_dedups_repeated_recommendations(self, mock_open: MagicMock) -> None:
        dup = {"paperId": "p1", "title": "T", "externalIds": {"ArXiv": "2402.1"}}
        mock_open.return_value = _resp(dup, dup)
        assert len(fetch_recommendations(["2106.09685"])) == 1
