"""Tests for the IACR ePrint source.

Two properties carry the weight here.

**The parse is tested against real captured markup**, not a hand-written mock. A fixture
invented to match the parser proves only that the parser matches itself; this one is a
verbatim slice of a live search response (`tests/fixtures/iacr_search_sample.html`), so if
ePrint's structure and this parser ever disagree, the disagreement is what fails.

**An unreadable page must never look like an empty one.** ePrint is scraped, so its markup
can change without warning. The adapter therefore distinguishes "no results" (ePrint says
so in its own words) from "zero parsed" (the structure moved), and raises on the second.
This project has already shipped one silent-empty failure — seven candidate pools cached
empty after an arXiv 429 storm and scored as honest zeros — and it cost a published number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reporadar.sources import iacr

FIXTURE = Path(__file__).parent / "fixtures" / "iacr_search_sample.html"


@pytest.fixture
def sample() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class TestParsesRealMarkup:
    def test_it_finds_every_result_in_the_captured_page(self, sample: str) -> None:
        papers = iacr.parse_results(sample, "side-channel")
        assert len(papers) == sample.count('class="paperlink"')
        assert len(papers) >= 2

    def test_each_paper_carries_what_the_gate_needs(self, sample: str) -> None:
        """C6: a record the pipeline cannot gate is a record that scores zero silently."""
        for paper in iacr.parse_results(sample, "q"):
            assert paper["arxiv_id"].startswith("iacr:")
            assert paper["title"].strip()
            assert paper["abstract"].strip(), "the 0-3 gate reads the abstract"
            assert paper["url"].startswith("https://eprint.iacr.org/")
            assert paper["source"] == "iacr"

    def test_ids_are_the_eprint_number_under_a_synthetic_prefix(self, sample: str) -> None:
        """Nothing downstream may try to resolve an ePrint id at arXiv."""
        for paper in iacr.parse_results(sample, "q"):
            number = paper["arxiv_id"].removeprefix("iacr:")
            assert number[:4].isdigit() and "/" in number

    def test_markup_is_stripped_from_titles_and_abstracts(self, sample: str) -> None:
        for paper in iacr.parse_results(sample, "q"):
            assert "<" not in paper["title"] and "<" not in paper["abstract"]
            assert "&amp;" not in paper["abstract"]

    def test_the_published_date_is_iso(self, sample: str) -> None:
        dated = [p for p in iacr.parse_results(sample, "q") if p["published"]]
        assert dated, "the captured page carries 'Last updated' dates"
        for paper in dated:
            assert paper["published"].endswith("T00:00:00Z")


class TestUnreadableIsNotEmpty:
    def test_a_changed_page_raises(self) -> None:
        with pytest.raises(iacr.CollectionError, match="structure"):
            iacr.parse_results("<html><body><p>something else entirely</p></body></html>", "q")

    def test_eprints_own_no_results_wording_returns_empty(self) -> None:
        assert iacr.parse_results("<h4>No results</h4>", "q") == []

    def test_the_marker_is_what_separates_them(self) -> None:
        """Both bodies parse to zero papers; only one is a failure."""
        body = "<div>lots of unfamiliar markup</div>"
        with pytest.raises(iacr.CollectionError):
            iacr.parse_results(body, "q")
        assert iacr.parse_results(body + iacr.NO_RESULTS_MARKER, "q") == []

    def test_a_network_failure_raises_rather_than_returning_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: Any, **_k: Any) -> Any:
            raise TimeoutError("slow")

        monkeypatch.setattr(iacr.urllib.request, "urlopen", boom)
        with pytest.raises(iacr.CollectionError):
            iacr._fetch("q")

    def test_collect_papers_logs_and_continues_past_one_bad_query(
        self, monkeypatch: pytest.MonkeyPatch, sample: str
    ) -> None:
        """One failing query must not silently zero the whole collection."""
        calls: list[str] = []

        def fake_fetch(query: str, timeout: int = 30) -> str:
            calls.append(query)
            if query == "bad":
                raise iacr.CollectionError("boom")
            return sample

        monkeypatch.setattr(iacr, "_fetch", fake_fetch)
        monkeypatch.setattr(iacr, "enrich_abstracts", lambda papers, **_k: 0)
        papers = iacr.collect_papers(["bad", "good"], lookback_days=36500)
        assert calls == ["bad", "good"]
        assert papers, "the surviving query's papers must still be returned"


class TestAbstractEnrichment:
    """The search page truncates at ~496 characters; the gate reads up to 1,500.

    Left alone, ePrint papers would reach the gate systematically less specific than arXiv
    ones, and a null result for this source could not be told apart from an artefact of
    its own adapter.
    """

    def test_a_longer_abstract_replaces_the_truncated_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        papers = [{"arxiv_id": "iacr:2026/1", "abstract": "short"}]
        monkeypatch.setattr(iacr, "fetch_full_abstract", lambda pid, **_k: "much longer text")
        assert iacr.enrich_abstracts(papers) == 1
        assert papers[0]["abstract"] == "much longer text"

    def test_it_never_shortens_an_abstract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        papers = [{"arxiv_id": "iacr:2026/1", "abstract": "a long existing abstract"}]
        monkeypatch.setattr(iacr, "fetch_full_abstract", lambda pid, **_k: "tiny")
        assert iacr.enrich_abstracts(papers) == 0
        assert papers[0]["abstract"] == "a long existing abstract"

    def test_a_failed_fetch_leaves_the_paper_intact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        papers = [{"arxiv_id": "iacr:2026/1", "abstract": "short"}]
        monkeypatch.setattr(iacr, "fetch_full_abstract", lambda pid, **_k: None)
        assert iacr.enrich_abstracts(papers) == 0
        assert papers[0]["abstract"] == "short"

    def test_papers_past_the_cap_are_kept_not_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shorter abstract is a handicap; a missing paper is a hole."""
        papers = [{"arxiv_id": f"iacr:2026/{i}", "abstract": "s"} for i in range(5)]
        monkeypatch.setattr(iacr, "fetch_full_abstract", lambda pid, **_k: "longer abstract")
        assert iacr.enrich_abstracts(papers, limit=2) == 2
        assert len(papers) == 5
        assert papers[4]["abstract"] == "s"


class TestPoliteness:
    def test_requests_go_through_a_process_wide_interval(self) -> None:
        """ePrint runs on donated infrastructure; the eval sweeps many repos per process."""
        assert iacr._MIN_REQUEST_INTERVAL_S >= 1.0
        source = Path(iacr.__file__).read_text(encoding="utf-8")
        assert source.count("_throttle()") >= 2, "both fetch paths must throttle"

    def test_the_user_agent_identifies_the_project(self) -> None:
        assert "RepoRadar" in iacr.USER_AGENT and "github.com" in iacr.USER_AGENT


class TestDateFiltering:
    def test_undated_papers_survive_the_cutoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ePrint sometimes omits the date; dropping those would quietly bias the source
        toward recently-touched work, which is the opposite of what it is here for."""
        body = '<a title="2026/1" class="paperlink" href="/2026/1">x</a><strong>T</strong>'
        body += '<p class="search-abstract">A</p>'
        monkeypatch.setattr(iacr, "_fetch", lambda q, timeout=30: body)
        monkeypatch.setattr(iacr, "enrich_abstracts", lambda papers, **_k: 0)
        papers = iacr.collect_papers(["q"], lookback_days=1)
        assert len(papers) == 1 and papers[0]["published"] == ""
