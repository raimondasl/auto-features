"""Tests for reporadar.digest."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from reporadar.digest import (
    MAYBE_THRESHOLD,
    TOP_THRESHOLD,
    categorize_papers,
    generate_digest,
    write_digest,
)
from reporadar.store import PaperStore


def _make_paper(arxiv_id: str = "2401.12345v1", **overrides) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "title": f"Test Paper {arxiv_id}",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We propose a novel approach to retrieval augmented generation.",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2024-01-20T00:00:00+00:00",
        "updated": "2024-01-21T00:00:00+00:00",
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
    }
    base.update(overrides)
    return base


def _make_score(arxiv_id: str, score_total: float, **overrides) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "score_total": score_total,
        "keyword_score": score_total * 0.5,
        "category_score": score_total * 0.3,
        "recency_score": score_total * 0.2,
        "matched_query": "all:test",
    }
    base.update(overrides)
    return base


def _seed_store(store: PaperStore) -> int:
    """Insert sample papers, record a run, and save scores. Returns run_id."""
    papers = [
        _make_paper("2401.00001v1", title="High Relevance RAG Paper"),
        _make_paper("2401.00002v1", title="Medium Relevance Transformers Paper"),
        _make_paper("2401.00003v1", title="Low Relevance Quantum Paper"),
        _make_paper("2401.00004v1", title="Barely Relevant Misc Paper"),
    ]
    store.upsert_papers(papers)
    run_id = store.record_run(["all:test", "all:retrieval"], papers_new=4, papers_seen=0)

    scores = [
        _make_score("2401.00001v1", 0.85),
        _make_score("2401.00002v1", 0.45),
        _make_score("2401.00003v1", 0.15),
        _make_score("2401.00004v1", 0.05),
    ]
    store.save_scores(run_id, scores)
    return run_id


class TestCategorizePapers:
    def test_splits_into_tiers(self) -> None:
        scored = [
            {"score_total": 0.9, "arxiv_id": "a"},
            {"score_total": 0.6, "arxiv_id": "b"},
            {"score_total": 0.3, "arxiv_id": "c"},
            {"score_total": 0.1, "arxiv_id": "d"},
        ]
        top, maybe, muted = categorize_papers(scored)

        assert len(top) == 2  # 0.9, 0.6 >= 0.5
        assert len(maybe) == 1  # 0.3 >= 0.2
        assert len(muted) == 1  # 0.1 < 0.2

    def test_respects_top_n(self) -> None:
        scored = [{"score_total": 0.9, "arxiv_id": f"p{i}"} for i in range(20)]
        top, maybe, muted = categorize_papers(scored, top_n=5)

        total = len(top) + len(maybe) + len(muted)
        assert total == 5

    def test_empty_input(self) -> None:
        top, maybe, muted = categorize_papers([])
        assert top == []
        assert maybe == []
        assert muted == []

    def test_all_top_picks(self) -> None:
        scored = [{"score_total": 0.8, "arxiv_id": f"p{i}"} for i in range(3)]
        top, maybe, muted = categorize_papers(scored)
        assert len(top) == 3
        assert len(maybe) == 0
        assert len(muted) == 0

    def test_all_muted(self) -> None:
        scored = [{"score_total": 0.05, "arxiv_id": f"p{i}"} for i in range(3)]
        top, maybe, muted = categorize_papers(scored)
        assert len(top) == 0
        assert len(maybe) == 0
        assert len(muted) == 3

    def test_custom_thresholds(self) -> None:
        scored = [
            {"score_total": 0.9, "arxiv_id": "a"},
            {"score_total": 0.5, "arxiv_id": "b"},
        ]
        top, maybe, muted = categorize_papers(
            scored, top_threshold=0.8, maybe_threshold=0.4
        )
        assert len(top) == 1
        assert len(maybe) == 1


class TestGenerateDigest:
    def test_renders_markdown(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "# RepoRadar Digest" in content
        assert "Top Picks" in content
        assert "High Relevance RAG Paper" in content

    def test_includes_maybe_section(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "Maybe Relevant" in content
        assert "Medium Relevance Transformers Paper" in content

    def test_includes_muted_section(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "Muted" in content

    def test_includes_score_breakdown(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "keyword:" in content
        assert "category:" in content
        assert "recency:" in content

    def test_includes_queries(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "all:test" in content
        assert "all:retrieval" in content

    def test_includes_arxiv_links(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id)

        assert "http://arxiv.org/abs/2401.00001v1" in content

    def test_empty_run(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = store.record_run(["q1"], 0, 0)
            content = generate_digest(store, run_id)

        assert "No scored papers found" in content

    def test_top_n_limits_output(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            content = generate_digest(store, run_id, top_n=1)

        # Only the highest-scoring paper should appear
        assert "High Relevance RAG Paper" in content
        assert "Medium Relevance" not in content
        assert "Low Relevance" not in content


class TestGenerateDigestSuggestions:
    def test_top_picks_have_suggestions_key(self, tmp_path: Path) -> None:
        """Top pick papers with matching abstract patterns should get suggestions."""
        with PaperStore(tmp_path / "papers.db") as store:
            # Insert a paper whose abstract triggers suggestion patterns
            paper = _make_paper(
                "2401.99999v1",
                title="Benchmark Paper",
                abstract="We evaluate on GLUE benchmark and outperforms BERT-base.",
            )
            store.upsert_paper(paper)
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.99999v1", 0.9)])

            content = generate_digest(store, run_id)

        assert "Action ideas" in content

    def test_suggestions_labeled_as_ideas(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            paper = _make_paper(
                "2401.99999v1",
                abstract="Code is open-sourced at our repository.",
            )
            store.upsert_paper(paper)
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(run_id, [_make_score("2401.99999v1", 0.9)])

            content = generate_digest(store, run_id)

        assert "auto-generated" in content


class TestWriteDigest:
    def test_writes_file(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out = write_digest(store, run_id, tmp_path / "output" / "digest.md")

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "# RepoRadar Digest" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out = write_digest(store, run_id, tmp_path / "deep" / "nested" / "digest.md")

        assert out.exists()

    def test_html_format(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out = write_digest(
                store, run_id, tmp_path / "digest.md", fmt="html"
            )

        assert out.suffix == ".html"
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "RepoRadar Digest" in content

    def test_html_format_explicit_extension(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed_store(store)
            out = write_digest(
                store, run_id, tmp_path / "output.html", fmt="html"
            )

        assert out.suffix == ".html"
        assert out.exists()
