"""Tests for reporadar.signals.hn."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from reporadar.signals.hn import (
    MAX_MONTHS_PER_RUN,
    MIN_POINTS,
    REFERENCE_POINTS,
    _stories_for,
    discussed_ids,
    fetch_attention,
    normalize_points,
)


def _resp(payload: object) -> MagicMock:
    mock = MagicMock()
    mock.read.return_value = json.dumps(payload).encode()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _story(url: str, points: int, comments: int = 0, object_id: str = "1") -> dict[str, Any]:
    return {
        "url": url,
        "points": points,
        "num_comments": comments,
        "objectID": object_id,
        "title": "A Story",
    }


class TestNormalizePoints:
    def test_scale_is_absolute_not_pool_relative(self) -> None:
        """The same point count must score the same in every run.

        This is the one place RepoRadar departs from the house pool-relative
        normalization, and it has to: a run typically contains 0-1 discussed papers,
        so the pool maximum *is* that paper and a relative scale would award 1.0 to a
        12-point story exactly as readily as to a 1300-point one.
        """
        alone = normalize_points({"a": 60})
        with_a_giant = normalize_points({"a": 60, "b": 5000})
        assert alone["a"] == with_a_giant["a"]
        assert with_a_giant["b"] == 1.0  # saturates at the reference, not the pool max

    def test_reference_points_saturate_at_one(self) -> None:
        scores = normalize_points({"a": REFERENCE_POINTS, "b": REFERENCE_POINTS * 10})
        assert scores["a"] == 1.0
        assert scores["b"] == 1.0

    def test_ordering_and_range(self) -> None:
        scores = normalize_points({"low": 15, "mid": 100, "high": 400})
        assert 0.0 < scores["low"] < scores["mid"] < scores["high"] <= 1.0

    def test_trivial_submissions_are_absent_not_low(self) -> None:
        # A paper posted and ignored is not worse than a paper nobody posted: a low
        # score would drag its total down below the never-posted paper's.
        scores = normalize_points({"ignored": MIN_POINTS - 1, "noticed": MIN_POINTS})
        assert "ignored" not in scores
        assert "noticed" in scores

    def test_zero_and_empty(self) -> None:
        assert normalize_points({}) == {}
        assert normalize_points({"a": 0}) == {}


class TestStoryFiltering:
    def test_typo_matches_are_rejected(self) -> None:
        # Algolia's typo tolerance returns a story about 2501.16948 when asked for
        # 2501.12948 — one digit off, and such a story can carry hundreds of points,
        # which would credit someone else's front-page discussion to this paper.
        hits = [
            _story("https://arxiv.org/abs/2501.16948", 358),
            _story("https://arxiv.org/abs/2501.12948", 1351),
        ]
        kept = _stories_for(hits, "2501.12948")
        assert [h["points"] for h in kept] == [1351]

    def test_hit_without_a_url_is_dropped(self) -> None:
        assert _stories_for([{"points": 10}], "2501.12948") == []


class TestFetchAttention:
    def _patch(self, payloads: list[object]) -> Any:
        return patch(
            "reporadar.signals.hn.urllib.request.urlopen",
            side_effect=[_resp(p) for p in payloads],
        )

    def test_month_prefilter_then_exact_confirm(self) -> None:
        month = {"hits": [_story("https://arxiv.org/abs/2501.12948", 1351)]}
        exact = {"hits": [_story("https://arxiv.org/abs/2501.12948", 1351, 1056, "42823568")]}
        with self._patch([month, exact]) as urlopen:
            got = fetch_attention(["2501.12948v1", "2501.99999v1"])
        # One request for the month, one to confirm the single candidate — not one
        # request per paper.
        assert urlopen.call_count == 2
        assert set(got) == {"2501.12948v1"}
        assert got["2501.12948v1"]["points"] == 1351
        assert got["2501.12948v1"]["story_url"].endswith("42823568")

    def test_papers_with_no_story_are_omitted(self) -> None:
        with self._patch([{"hits": []}]):
            assert fetch_attention(["2501.00001v1", "2501.00002v1"]) == {}

    def test_resubmissions_aggregate_to_the_best(self) -> None:
        month = {"hits": [_story("https://arxiv.org/abs/2402.17764", 2)]}
        exact = {
            "hits": [
                _story("https://arxiv.org/abs/2402.17764", 2, 0, "a"),
                _story("https://arxiv.org/abs/2402.17764", 1040, 447, "b"),
                _story("https://arxiv.org/abs/2402.17764", 12, 3, "c"),
            ]
        }
        with self._patch([month, exact]):
            got = fetch_attention(["2402.17764v1"])
        assert got["2402.17764v1"]["points"] == 1040
        assert got["2402.17764v1"]["submissions"] == 3

    def test_non_arxiv_and_legacy_ids_are_skipped(self) -> None:
        with patch("reporadar.signals.hn.urllib.request.urlopen") as urlopen:
            assert fetch_attention(["ss:1", "oa:W2", "math/0309395"]) == {}
            urlopen.assert_not_called()

    def test_network_failure_degrades_to_no_signal(self) -> None:
        with patch(
            "reporadar.signals.hn.urllib.request.urlopen", side_effect=OSError("unreachable")
        ):
            assert fetch_attention(["2501.12948v1"]) == {}

    def test_unparseable_response_degrades(self) -> None:
        broken = MagicMock()
        broken.read.return_value = b"<html>not json</html>"
        broken.__enter__ = MagicMock(return_value=broken)
        broken.__exit__ = MagicMock(return_value=False)
        with patch("reporadar.signals.hn.urllib.request.urlopen", return_value=broken):
            assert fetch_attention(["2501.12948v1"]) == {}

    def test_empty_input_makes_no_requests(self) -> None:
        with patch("reporadar.signals.hn.urllib.request.urlopen") as urlopen:
            assert fetch_attention([]) == {}
            urlopen.assert_not_called()

    def test_request_uses_the_safe_query_form(self) -> None:
        with patch(
            "reporadar.signals.hn.urllib.request.urlopen", return_value=_resp({"hits": []})
        ) as urlopen:
            discussed_ids(["2501.12948v1"])
        url = urlopen.call_args[0][0].full_url
        # Without restrictSearchableAttributes=url a bare id matches ~115 unrelated
        # "Show HN: an arXiv tool" stories; without tags=story the hits are comments
        # whose points/url are null; typoTolerance guards the wrong-paper case.
        assert "restrictSearchableAttributes=url" in url
        assert "tags=story" in url
        assert "typoTolerance=false" in url

    def test_truncated_month_sweep_is_reported(self, caplog: Any) -> None:
        # Algolia caps a query at 1000 hits and still reports nbPages: 1 with HTTP
        # 200, so nbPages cannot detect truncation — only nbHits > len(hits) can.
        payload = {"hits": [_story("https://arxiv.org/abs/2607.00001", 5)], "nbHits": 2461}
        with patch("reporadar.signals.hn.urllib.request.urlopen", return_value=_resp(payload)):
            discussed_ids(["2607.00001v1"])
        assert any("may be missed" in r.getMessage() for r in caplog.records)

    def test_complete_sweep_is_not_reported(self, caplog: Any) -> None:
        payload = {"hits": [_story("https://arxiv.org/abs/2607.00001", 5)], "nbHits": 1}
        with patch("reporadar.signals.hn.urllib.request.urlopen", return_value=_resp(payload)):
            discussed_ids(["2607.00001v1"])
        assert not any("may be missed" in r.getMessage() for r in caplog.records)

    def test_month_sweep_is_capped(self, caplog: Any) -> None:
        # `--foundational` reaches back over all of arXiv; one request per month would
        # be hundreds against an API that enforces limits by blacklisting the IP.
        ids = [f"{yy:02d}{mm:02d}.00001v1" for yy in range(15, 27) for mm in range(1, 13)]
        with patch(
            "reporadar.signals.hn.urllib.request.urlopen", return_value=_resp({"hits": []})
        ) as urlopen:
            discussed_ids(ids)
        assert urlopen.call_count == MAX_MONTHS_PER_RUN
        assert any("limited to the" in r.getMessage() for r in caplog.records)

    def test_one_request_per_distinct_month(self) -> None:
        with patch(
            "reporadar.signals.hn.urllib.request.urlopen", return_value=_resp({"hits": []})
        ) as urlopen:
            discussed_ids(["2501.00001v1", "2501.00002v1", "2607.00003v1"])
        assert urlopen.call_count == 2
