"""Tests for reporadar.provenance — which channel a paper came from, and who records it.

The guards matter more than the mapping. The channel is recorded in one column by four
different writers, and a reader that silently files an unknown marker under "arXiv keyword
search" would report dense discovery's contributions as keyword search's -- the exact
confusion this module exists to remove.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from reporadar import pipeline, provenance
from reporadar.provenance import describe, found_by, source_marker

SRC = Path(__file__).resolve().parents[1] / "src" / "reporadar"


class TestFoundBy:
    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            ("hyde", "dense_discovery"),
            ("recommendation", "s2_recommendations"),
            ("source:openalex", "openalex"),
            ("source:semantic_scholar", "semantic_scholar"),
            ('all:"retrieval augmented generation" AND cat:cs.CL', "arxiv_keywords"),
            ("ti:transformer", "arxiv_keywords"),
            (None, "unrecorded"),
            ("", "unrecorded"),
            ("source:", "unrecorded"),
        ],
    )
    def test_each_marker_reads_back_as_its_channel(self, stored: str | None, expected: str) -> None:
        assert found_by(stored) == expected

    def test_a_source_marker_round_trips(self) -> None:
        for key in provenance.SOURCE_LABELS:
            assert found_by(source_marker(key)) == key


class TestDescribe:
    def test_dense_discovery_says_keyword_search_did_not_have_it(self) -> None:
        """That clause is the point: it is what makes the provenance worth reporting."""
        text = describe("hyde")
        assert "HyDE" in text and "keyword search did not find it" in text

    def test_keyword_search_names_the_query_that_matched(self) -> None:
        assert "all:transformers" in describe("all:transformers")

    def test_a_source_is_named_the_way_users_know_it(self) -> None:
        assert describe(source_marker("europepmc")) == "Europe PMC"

    def test_an_unrecorded_paper_says_why_rather_than_guessing(self) -> None:
        assert "not recorded" in describe(None)


class TestNoWriterGoesUnrecognised:
    """The drift guards."""

    def test_every_literal_marker_in_the_source_is_recognised(self) -> None:
        """A writer that invents a new `matched_query` value would otherwise be reported as
        arXiv keyword search -- plausible-looking, and wrong."""
        found: dict[str, str] = {}
        for path in SRC.rglob("*.py"):
            body = path.read_text(encoding="utf-8")
            for literal in re.findall(r"""["']matched_query["']\s*:\s*["']([^"']+)["']""", body):
                found[literal] = path.name
        assert found, "no literal matched_query writers found; the pattern is out of date"
        misread = {
            lit: where
            for lit, where in found.items()
            if found_by(lit) in {"arxiv_keywords", "unrecorded"}
        }
        assert not misread, (
            f"these markers would read back as keyword search or unrecorded: {misread}; "
            "teach reporadar.provenance about them"
        )

    def test_every_keyword_source_has_a_label(self) -> None:
        """`KEYWORD_SOURCES` is what actually runs, so it is the list to hold labels to."""
        assert set(provenance.SOURCE_LABELS) == set(pipeline.KEYWORD_SOURCES)


class _Quiet:
    def info(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...


def _paper(arxiv_id: str, **extra: Any) -> dict[str, Any]:
    return {"arxiv_id": arxiv_id, "title": arxiv_id, **extra}


class TestMergedSourcesAreStamped:
    def test_a_paper_a_source_contributes_records_that_source(self) -> None:
        papers = [_paper("2401.00001", matched_query="all:x")]
        pipeline._merge_source(
            papers, "OpenAlex", lambda: [_paper("2402.00002")], source="openalex", report=_Quiet()
        )
        added = papers[-1]
        assert added["arxiv_id"] == "2402.00002"
        assert found_by(added["matched_query"]) == "openalex"

    def test_a_paper_already_found_keeps_the_channel_that_found_it_first(self) -> None:
        """Version-insensitively: the same paper under `v2` is not a new contribution."""
        papers = [_paper("2401.00001", matched_query="hyde")]
        pipeline._merge_source(
            papers, "DBLP", lambda: [_paper("2401.00001v2")], source="dblp", report=_Quiet()
        )
        assert len(papers) == 1
        assert found_by(papers[0]["matched_query"]) == "dense_discovery"

    def test_a_marker_the_source_set_itself_is_not_overwritten(self) -> None:
        papers: list[dict[str, Any]] = []
        pipeline._merge_source(
            papers,
            "Semantic Scholar",
            lambda: [_paper("2403.00003", matched_query="recommendation")],
            source="semantic_scholar",
            report=_Quiet(),
        )
        assert found_by(papers[0]["matched_query"]) == "s2_recommendations"

    def test_the_fetched_objects_are_not_mutated(self) -> None:
        """Stamped copies, not the source's own dicts: a fetcher may hand back cached objects,
        and stamping them in place would leak one run's provenance into the next."""
        fetched = [_paper("2404.00004")]
        pipeline._merge_source([], "bioRxiv", lambda: fetched, source="biorxiv", report=_Quiet())
        assert "matched_query" not in fetched[0]
