"""Tests for reporadar.trends."""

from __future__ import annotations

from pathlib import Path

from reporadar.profiler import RepoProfile
from reporadar.store import PaperStore
from reporadar.trends import compute_keyword_frequencies, detect_trends


def _make_profile(**overrides) -> RepoProfile:
    defaults = {
        "keywords": [("transformer", 0.8), ("attention", 0.6), ("retrieval", 0.4)],
        "anchors": ["torch"],
        "domains": ["deep learning"],
    }
    defaults.update(overrides)
    return RepoProfile(**defaults)


def _make_paper(arxiv_id: str, title: str = "", abstract: str = "") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["Alice"],
        "abstract": abstract,
        "categories": ["cs.CL"],
        "published": "2024-01-20T00:00:00+00:00",
        "url": f"http://arxiv.org/abs/{arxiv_id}",
    }


class TestComputeKeywordFrequencies:
    def test_counts_matches(self) -> None:
        profile = _make_profile()
        papers = [
            _make_paper("1", abstract="A new transformer model"),
            _make_paper("2", abstract="Attention is all you need"),
            _make_paper("3", abstract="A transformer with attention"),
        ]
        freqs = compute_keyword_frequencies(papers, profile)
        assert freqs["transformer"] == 2
        assert freqs["attention"] == 2
        assert freqs["retrieval"] == 0

    def test_empty_papers(self) -> None:
        profile = _make_profile()
        freqs = compute_keyword_frequencies([], profile)
        assert all(v == 0 for v in freqs.values())

    def test_empty_keywords(self) -> None:
        profile = _make_profile(keywords=[])
        papers = [_make_paper("1", abstract="Some text")]
        freqs = compute_keyword_frequencies(papers, profile)
        assert freqs == {}


class TestDetectTrends:
    def test_detects_increase(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            # Create 3 previous runs with low frequencies
            for i in range(3):
                rid = store.record_run([f"q{i}"], 1, 0)
                store.save_keyword_frequencies(rid, {"transformer": 2, "attention": 1})

            # Current run with much higher frequency
            current_rid = store.record_run(["q_current"], 1, 0)
            store.save_keyword_frequencies(current_rid, {"transformer": 10, "attention": 1})

            trends = detect_trends(store, current_rid, lookback_runs=3)

        assert len(trends) >= 1
        t_keywords = [t["keyword"] for t in trends]
        assert "transformer" in t_keywords
        # attention didn't change so shouldn't be trending
        assert "attention" not in t_keywords

    def test_no_previous_runs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            rid = store.record_run(["q1"], 1, 0)
            store.save_keyword_frequencies(rid, {"transformer": 5})
            trends = detect_trends(store, rid)
        assert trends == []

    def test_no_frequencies(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            rid = store.record_run(["q1"], 1, 0)
            trends = detect_trends(store, rid)
        assert trends == []

    def test_small_increase_not_trending(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            for i in range(3):
                rid = store.record_run([f"q{i}"], 1, 0)
                store.save_keyword_frequencies(rid, {"transformer": 10})

            current_rid = store.record_run(["q_current"], 1, 0)
            # Only 10% increase — should not be detected
            store.save_keyword_frequencies(current_rid, {"transformer": 11})

            trends = detect_trends(store, current_rid, lookback_runs=3)

        assert len(trends) == 0
