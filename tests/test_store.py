"""Tests for reporadar.store."""

from __future__ import annotations

from pathlib import Path

import pytest

from reporadar.store import PaperStore


def _make_paper(**overrides) -> dict:
    """Create a sample paper dict with sensible defaults."""
    base = {
        "arxiv_id": "2401.12345v1",
        "title": "Test Paper on Retrieval Augmented Generation",
        "authors": ["Alice Smith", "Bob Jones"],
        "abstract": "We propose a novel approach to RAG using long context transformers.",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2024-01-20T00:00:00+00:00",
        "updated": "2024-01-21T00:00:00+00:00",
        "url": "http://arxiv.org/abs/2401.12345v1",
        "pdf_url": "http://arxiv.org/pdf/2401.12345v1",
    }
    base.update(overrides)
    return base


class TestPaperStoreInit:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sub" / "papers.db"
        with PaperStore(db_path):
            pass
        assert db_path.exists()

    def test_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "papers.db"
        with PaperStore(db_path) as store:
            tables = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = [t["name"] for t in tables]

        assert "papers" in table_names
        assert "runs" in table_names
        assert "paper_scores" in table_names

    def test_idempotent_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "papers.db"
        with PaperStore(db_path) as store:
            store.upsert_paper(_make_paper())
        # Re-open — should not lose data
        with PaperStore(db_path) as store:
            assert store.paper_count() == 1


class TestUpsertPaper:
    def test_insert_new_paper(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            is_new = store.upsert_paper(_make_paper())
            assert is_new is True
            assert store.paper_count() == 1

    def test_update_existing_paper(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper())
            is_new = store.upsert_paper(_make_paper(title="Updated Title"))
            assert is_new is False
            assert store.paper_count() == 1

            paper = store.get_paper("2401.12345v1")
            assert paper is not None
            assert paper["title"] == "Updated Title"

    def test_different_ids_are_separate(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.11111v1"))
            store.upsert_paper(_make_paper(arxiv_id="2401.22222v1"))
            assert store.paper_count() == 2


class TestUpsertPapers:
    def test_batch_insert(self, tmp_path: Path) -> None:
        papers = [
            _make_paper(arxiv_id="2401.00001v1"),
            _make_paper(arxiv_id="2401.00002v1"),
            _make_paper(arxiv_id="2401.00003v1"),
        ]
        with PaperStore(tmp_path / "papers.db") as store:
            new, seen = store.upsert_papers(papers)
            assert new == 3
            assert seen == 0

    def test_batch_mixed(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))

            papers = [
                _make_paper(arxiv_id="2401.00001v1"),  # existing
                _make_paper(arxiv_id="2401.00002v1"),  # new
            ]
            new, seen = store.upsert_papers(papers)
            assert new == 1
            assert seen == 1


class TestGetPaper:
    def test_get_existing(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper())
            paper = store.get_paper("2401.12345v1")

        assert paper is not None
        assert paper["title"] == "Test Paper on Retrieval Augmented Generation"
        assert paper["authors"] == ["Alice Smith", "Bob Jones"]
        assert paper["categories"] == ["cs.CL", "cs.LG"]

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            assert store.get_paper("9999.99999v1") is None


class TestGetAllPapers:
    def test_returns_all(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_papers([
                _make_paper(arxiv_id="2401.00001v1"),
                _make_paper(arxiv_id="2401.00002v1"),
            ])
            papers = store.get_all_papers()
        assert len(papers) == 2

    def test_empty_db(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            assert store.get_all_papers() == []


class TestRecordRun:
    def test_records_and_retrieves(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = store.record_run(
                queries_used=["all:transformers", "all:RAG"],
                papers_new=5,
                papers_seen=2,
            )
            assert run_id is not None

            runs = store.get_runs()
            assert len(runs) == 1
            assert runs[0]["papers_new"] == 5
            assert runs[0]["papers_seen"] == 2
            assert runs[0]["queries_used"] == ["all:transformers", "all:RAG"]

    def test_get_last_run(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.record_run(["q1"], 1, 0)
            store.record_run(["q2"], 2, 1)

            last = store.get_last_run()
            assert last is not None
            assert last["papers_new"] == 2

    def test_no_runs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            assert store.get_last_run() is None


class TestScores:
    def test_save_and_retrieve(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))
            store.upsert_paper(_make_paper(arxiv_id="2401.00002v1"))
            run_id = store.record_run(["q1"], 2, 0)

            scores = [
                {
                    "arxiv_id": "2401.00001v1",
                    "score_total": 0.85,
                    "keyword_score": 0.5,
                    "category_score": 0.2,
                    "recency_score": 0.15,
                    "matched_query": "q1",
                },
                {
                    "arxiv_id": "2401.00002v1",
                    "score_total": 0.42,
                    "keyword_score": 0.3,
                    "category_score": 0.1,
                    "recency_score": 0.02,
                    "matched_query": "q1",
                },
            ]
            store.save_scores(run_id, scores)

            retrieved = store.get_scores_for_run(run_id)
            assert len(retrieved) == 2
            # Should be ordered by score descending
            assert retrieved[0]["score_total"] == 0.85
            assert retrieved[1]["score_total"] == 0.42
            # Should include joined paper data
            assert "title" in retrieved[0]
            assert "authors" in retrieved[0]
