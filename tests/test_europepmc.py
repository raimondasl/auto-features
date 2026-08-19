"""Tests for reporadar.sources.europepmc (mocked HTTP).

The adapter exists because bioRxiv's own API cannot be searched by keyword, and it was
written after probing the live Europe PMC API rather than from its documentation. The cases
below are the probe's findings turned into guards — every literal string here was taken from
a real response on 2026-08-19, and the three that matter most (`p < 0.001`, unquoted
keywords, markup in titles) are each a defect the documentation would not have revealed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from reporadar.sources.europepmc import (
    EuropePMCError,
    _strip_markup,
    build_query,
    collect_papers,
    search_papers,
)


def _record(doi: str = "10.64898/2026.07.30.741827", **overrides) -> dict:
    base = {
        "id": "PPR1291877",
        "source": "PPR",
        "doi": doi,
        "title": "PlantNetX: a transcriptomic resource",
        "authorString": "Nasir MA, Nawaz S, Faik A.",
        "abstractText": "Bulk RNA sequencing provides complementary information.",
        "firstPublicationDate": "2026-08-05",
        "pubYear": "2026",
        # Both null in every sampled response, despite `PUBLISHER:` filtering correctly.
        "publisher": None,
        "pubType": None,
    }
    base.update(overrides)
    return base


def _response(*records: dict, hit_count: int | None = None) -> dict:
    return {
        "hitCount": len(records) if hit_count is None else hit_count,
        "resultList": {"result": list(records)},
    }


class TestTheQueryIsTheOneThatWasMeasured:
    def test_keywords_are_not_quoted(self) -> None:
        """The single most consequential line in the adapter.

        These strings come from `collector.to_plain_keywords` and are bags of words, not
        phrases. Quoting makes Europe PMC demand the exact phrase: measured over eight
        product-shaped queries it returned 0 or 1 hits for four of them against 85-2,239
        unquoted — `sequence alignment long reads` was 0 vs 2,239.
        """
        query = build_query("sequence alignment long reads", 14)
        assert "(sequence alignment long reads)" in query
        assert '"sequence alignment long reads"' not in query

    def test_it_asks_for_preprints_from_the_two_named_servers(self) -> None:
        query = build_query("crispr screen", 14)
        assert "SRC:PPR" in query
        assert 'PUBLISHER:"bioRxiv"' in query
        assert 'PUBLISHER:"medRxiv"' in query

    def test_a_bounded_window_becomes_a_date_clause(self) -> None:
        assert "FIRST_PDATE:[" in build_query("crispr screen", 14)

    def test_an_all_time_lookback_sends_no_date_clause(self) -> None:
        """The measured configuration runs `lookback_days: 36500`. Asking for everything
        since 1926 is a slower way of asking for everything."""
        assert "FIRST_PDATE" not in build_query("crispr screen", 36500)


class TestMarkupIsStrippedWithoutEatingTheScience:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Mapping  <i>trans</i>  -eQTLs", "Mapping trans -eQTLs"),
            ("TREVI  <sup>XMBD</sup>  : a model", "TREVI XMBD : a model"),
            ("Result. <h4>Availability</h4> Source at git.", "Result. Availability Source at git."),
            ("a <sub>2</sub> b", "a 2 b"),
        ],
    )
    def test_real_tags_go(self, raw: str, expected: str) -> None:
        """18% of 785 sampled titles carried `<i>` or `<sup>`; 36% of abstracts carried
        `<h4>`. The spec only mentioned abstracts. Whitespace is collapsed because the tags
        are padded — 91 of 400 titles were left with a double space by tag removal alone."""
        assert _strip_markup(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "discrimination (t = 14.80, p < 0.001, Cohen's d = 1.64), scaling to d = 8.96.",
            "we detect <Choloepus didactylus> and <Tamandua tetradactyla> genomes",
            "<<1% have tri-kinetochores, which we confirm with long-read CENP-A",
            "reduced from <1% Wolbachia reads to 90% under adaptive sampling",
            "isolates as multidrug resistant (p < 0.0001). Importantly, key AMR genes",
        ],
    )
    def test_the_science_stays(self, raw: str) -> None:
        """The paired guard, and the reason the pattern requires a tag NAME.

        `<[^>]+>` is the obvious pattern and it mutilates biology abstracts: `p < 0.001`
        opens a span that the regex closes at the NEXT real tag, taking everything between.
        On 785 sampled records that removed 3,910 characters of real abstract — one example
        lost the whole results sentence, 240 characters, between `p ` and `Availability`.
        These five strings are the exact spans it would have eaten.
        """
        assert _strip_markup(raw) == raw


class TestNormalisation:
    @patch("reporadar.sources.europepmc._request_json")
    def test_the_doi_is_the_id(self, mock_req: MagicMock) -> None:
        """So a preprint arriving here and from OpenAlex is one paper, not two (F15)."""
        mock_req.return_value = _response(_record())
        [paper] = search_papers("crispr", lookback_days=14)
        assert paper["arxiv_id"] == "doi:10.64898/2026.07.30.741827"
        assert paper["url"] == "https://doi.org/10.64898/2026.07.30.741827"

    @patch("reporadar.sources.europepmc._request_json")
    def test_both_doi_prefixes_are_handled(self, mock_req: MagicMock) -> None:
        """10.1101 is bioRxiv's original prefix and 10.64898 its current one; a 785-record
        sample held 284 of the first and 216 of the second, so both are live."""
        mock_req.return_value = _response(
            _record("10.1101/2022.01.02.474000"), _record("10.64898/2026.07.30.741827")
        )
        papers = search_papers("crispr", lookback_days=14)
        assert [p["arxiv_id"] for p in papers] == [
            "doi:10.1101/2022.01.02.474000",
            "doi:10.64898/2026.07.30.741827",
        ]

    @patch("reporadar.sources.europepmc._request_json")
    def test_categories_are_left_empty(self, mock_req: MagicMock) -> None:
        """Europe PMC returns no subject classification for a preprint — `publisher` and
        `pubType` are both null. Filling `categories` from another taxonomy is exactly the
        F4 defect; an empty list takes `ranking.absent_category`, the policy for this case.
        """
        mock_req.return_value = _response(_record())
        [paper] = search_papers("crispr", lookback_days=14)
        assert paper["categories"] == []

    @patch("reporadar.sources.europepmc._request_json")
    def test_a_record_without_a_usable_doi_is_dropped(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _response(
            _record(doi=""), _record(doi="not-a-doi"), _record("10.1101/ok.1")
        )
        assert [p["arxiv_id"] for p in search_papers("x", 14)] == ["doi:10.1101/ok.1"]

    @patch("reporadar.sources.europepmc._request_json")
    def test_the_title_is_stripped_as_well_as_the_abstract(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _response(
            _record(
                title="Mapping  <i>trans</i>  -eQTLs",
                abstractText="Findings. <h4>Availability</h4> At github.",
            )
        )
        [paper] = search_papers("x", 14)
        assert paper["title"] == "Mapping trans -eQTLs"
        assert paper["abstract"] == "Findings. Availability At github."


class TestARefusalIsNotAZero:
    """The mistake this project has published twice: an API that said no, recorded as a
    source that found nothing."""

    @patch("reporadar.sources.europepmc._request_json")
    def test_an_empty_result_is_an_empty_list(self, mock_req: MagicMock) -> None:
        """Europe PMC answers a genuine miss with `hitCount: 0` and a present `resultList`,
        so honest emptiness is distinguishable from refusal at the transport level."""
        mock_req.return_value = _response(hit_count=0)
        assert search_papers("zzqx nonexistent qqz", 14) == []

    @patch("reporadar.sources.europepmc._request_json")
    def test_one_refused_query_is_skipped_not_fatal(self, mock_req: MagicMock) -> None:
        mock_req.side_effect = [EuropePMCError("HTTP 503"), _response(_record())]
        papers = collect_papers(["a", "b"], lookback_days=36500)
        assert len(papers) == 1

    @patch("reporadar.sources.europepmc._request_json")
    def test_every_query_refused_raises(self, mock_req: MagicMock) -> None:
        """Returning `[]` here would let a caller record "Europe PMC contributed nothing"
        about a conversation that never happened."""
        mock_req.side_effect = EuropePMCError("HTTP 503")
        with pytest.raises(EuropePMCError, match="not a zero"):
            collect_papers(["a", "b"], lookback_days=36500)

    @patch("reporadar.sources.europepmc.urllib.request.urlopen")
    @patch("reporadar.sources.europepmc.time.sleep")
    def test_a_4xx_is_not_retried(self, _sleep: MagicMock, mock_open: MagicMock) -> None:
        import urllib.error

        mock_open.side_effect = urllib.error.HTTPError("u", 400, "Bad Request", {}, None)
        with pytest.raises(EuropePMCError, match="400"):
            search_papers("x", 14)
        assert mock_open.call_count == 1

    @patch("reporadar.sources.europepmc.urllib.request.urlopen")
    @patch("reporadar.sources.europepmc.time.sleep")
    def test_a_503_is_retried_then_succeeds(self, _sleep: MagicMock, mock_open: MagicMock) -> None:
        """A burst of unspaced requests drew 504s and then a 503 during probing; 22
        consecutive spaced ones completed clean. The flakiness is real and transient."""
        import urllib.error

        ok = MagicMock()
        ok.read.return_value = json.dumps(_response(_record())).encode()
        ok.__enter__ = lambda s: s
        ok.__exit__ = lambda *a: False
        mock_open.side_effect = [
            urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None),
            ok,
        ]
        assert len(search_papers("x", 14)) == 1


class TestCollect:
    @patch("reporadar.sources.europepmc.time.sleep")
    @patch("reporadar.sources.europepmc._request_json")
    def test_queries_are_merged_and_deduplicated(
        self, mock_req: MagicMock, _sleep: MagicMock
    ) -> None:
        mock_req.side_effect = [
            _response(_record("10.1101/a"), _record("10.1101/b")),
            _response(_record("10.1101/b"), _record("10.1101/c")),
        ]
        papers = collect_papers(["one", "two"], lookback_days=36500)
        assert sorted(p["arxiv_id"] for p in papers) == [
            "doi:10.1101/a",
            "doi:10.1101/b",
            "doi:10.1101/c",
        ]

    @patch("reporadar.sources.europepmc._request_json")
    def test_blank_queries_cost_no_request(self, mock_req: MagicMock) -> None:
        assert collect_papers(["", "   "], lookback_days=36500) == []
        assert mock_req.call_count == 0
