"""Tests for reporadar.citation_graph — seed set + citation-link logic."""

from __future__ import annotations

from pathlib import Path

from reporadar.citation_graph import base_id, build_seed_set, find_citation_links
from reporadar.store import PaperStore


class TestBaseId:
    def test_strips_version(self) -> None:
        assert base_id("2401.00001v2") == "2401.00001"
        assert base_id("2401.00001") == "2401.00001"


class TestBuildSeedSet:
    def test_starred_and_high_ratings_only(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.star_paper("2401.1v1")
            store.save_rating("2401.2v1", 5)  # >= 4 → seed
            store.save_rating("2401.3v1", 4)  # >= 4 → seed
            store.save_rating("2401.4v1", 2)  # < 4 → not a seed
            seeds = build_seed_set(store)
        assert seeds == {"2401.1", "2401.2", "2401.3"}

    def test_empty(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            assert build_seed_set(store) == set()

    def test_custom_min_rating(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.save_rating("2401.2v1", 3)
            assert build_seed_set(store, min_rating=3) == {"2401.2"}
            assert build_seed_set(store, min_rating=4) == set()


class TestFindCitationLinks:
    def test_matches_version_insensitive(self) -> None:
        refs = {"A": ["2401.00099v1", "9999.9"], "B": ["1234.5"]}
        assert find_citation_links(refs, {"2401.00099"}) == {"A": ["2401.00099"]}

    def test_multiple_seed_hits_sorted(self) -> None:
        refs = {"A": ["2401.00300", "2401.00100"]}
        assert find_citation_links(refs, {"2401.00100", "2401.00300"}) == {
            "A": ["2401.00100", "2401.00300"]
        }

    def test_no_seeds_is_empty(self) -> None:
        assert find_citation_links({"A": ["x"]}, set()) == {}

    def test_no_hits_is_empty(self) -> None:
        assert find_citation_links({"A": ["9999.9"]}, {"2401.1"}) == {}
