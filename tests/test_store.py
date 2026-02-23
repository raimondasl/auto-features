"""Tests for reporadar.store."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest

from reporadar.store import CURRENT_SCHEMA_VERSION, PaperStore, StoreError


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


class TestCorruptDb:
    def test_corrupt_file_raises_store_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "corrupt.db"
        db_path.write_bytes(b"this is not a valid sqlite database!!")

        with pytest.raises(StoreError, match="Cannot open database"):
            PaperStore(db_path)


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
            store.upsert_papers(
                [
                    _make_paper(arxiv_id="2401.00001v1"),
                    _make_paper(arxiv_id="2401.00002v1"),
                ]
            )
            papers = store.get_all_papers()
        assert len(papers) == 2

    def test_empty_db(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            assert store.get_all_papers() == []


class TestGetPapersSince:
    def test_filters_by_first_seen(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))

            # Papers inserted now should be found when searching from an hour ago
            from datetime import datetime, timedelta

            one_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            papers = store.get_papers_since(one_hour_ago)
            assert len(papers) == 1
            assert papers[0]["arxiv_id"] == "2401.00001v1"

    def test_excludes_old_papers(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))

            from datetime import datetime, timedelta

            future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
            papers = store.get_papers_since(future)
            assert len(papers) == 0

    def test_empty_db(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            papers = store.get_papers_since("2020-01-01T00:00:00+00:00")
            assert papers == []


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

    def test_get_runs_with_limit(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            for i in range(5):
                store.record_run([f"q{i}"], i, 0)

            all_runs = store.get_runs()
            assert len(all_runs) == 5

            limited = store.get_runs(limit=2)
            assert len(limited) == 2
            # Should be most recent first
            assert limited[0]["papers_new"] == 4
            assert limited[1]["papers_new"] == 3


class TestPreviousRunAndScoredIds:
    def test_get_previous_run_id(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            r1 = store.record_run(["q1"], 1, 0)
            r2 = store.record_run(["q2"], 2, 0)
            r3 = store.record_run(["q3"], 3, 0)

            assert store.get_previous_run_id(r3) == r2
            assert store.get_previous_run_id(r2) == r1
            assert store.get_previous_run_id(r1) is None

    def test_get_scored_paper_ids_for_run(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))
            store.upsert_paper(_make_paper(arxiv_id="2401.00002v1"))
            run_id = store.record_run(["q1"], 2, 0)
            store.save_scores(
                run_id,
                [
                    {
                        "arxiv_id": "2401.00001v1",
                        "score_total": 0.8,
                        "keyword_score": 0.5,
                        "category_score": 0.2,
                        "recency_score": 0.1,
                    },
                    {
                        "arxiv_id": "2401.00002v1",
                        "score_total": 0.4,
                        "keyword_score": 0.3,
                        "category_score": 0.1,
                        "recency_score": 0.0,
                    },
                ],
            )

            ids = store.get_scored_paper_ids_for_run(run_id)
            assert ids == {"2401.00001v1", "2401.00002v1"}

    def test_get_scored_paper_ids_empty(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = store.record_run(["q1"], 0, 0)
            ids = store.get_scored_paper_ids_for_run(run_id)
            assert ids == set()


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


class TestSchemaVersion:
    def test_fresh_db_has_correct_version(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            assert store.schema_version() == CURRENT_SCHEMA_VERSION

    def test_reopened_db_skips_migration(self, tmp_path: Path) -> None:
        db_path = tmp_path / "papers.db"
        with PaperStore(db_path) as store:
            store.upsert_paper(_make_paper())

        with PaperStore(db_path) as store:
            assert store.schema_version() == CURRENT_SCHEMA_VERSION
            assert store.paper_count() == 1

    def test_legacy_db_gets_bootstrapped(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "legacy.db"
        # Create a legacy DB with tables but no schema_version
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""\
            CREATE TABLE papers (
                arxiv_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                authors TEXT NOT NULL, abstract TEXT NOT NULL,
                categories TEXT NOT NULL, published TEXT NOT NULL,
                updated TEXT, url TEXT NOT NULL, pdf_url TEXT,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
            );
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time TEXT NOT NULL, queries_used TEXT NOT NULL,
                papers_new INTEGER NOT NULL, papers_seen INTEGER NOT NULL
            );
            CREATE TABLE paper_scores (
                arxiv_id TEXT NOT NULL, run_id INTEGER NOT NULL,
                score_total REAL NOT NULL, keyword_score REAL,
                category_score REAL, recency_score REAL,
                matched_query TEXT, PRIMARY KEY (arxiv_id, run_id)
            );
        """)
        conn.close()

        with PaperStore(db_path) as store:
            assert store.schema_version() == CURRENT_SCHEMA_VERSION


class TestSchemaMigrationV3:
    def test_v2_db_gets_migrated(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "v2.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""\
            CREATE TABLE papers (
                arxiv_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                authors TEXT NOT NULL, abstract TEXT NOT NULL,
                categories TEXT NOT NULL, published TEXT NOT NULL,
                updated TEXT, url TEXT NOT NULL, pdf_url TEXT,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
            );
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time TEXT NOT NULL, queries_used TEXT NOT NULL,
                papers_new INTEGER NOT NULL, papers_seen INTEGER NOT NULL
            );
            CREATE TABLE paper_scores (
                arxiv_id TEXT NOT NULL, run_id INTEGER NOT NULL,
                score_total REAL NOT NULL, keyword_score REAL,
                category_score REAL, recency_score REAL,
                embedding_score REAL, citation_score REAL,
                matched_query TEXT, PRIMARY KEY (arxiv_id, run_id)
            );
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (2);
        """)
        conn.close()

        with PaperStore(db_path) as store:
            assert store.schema_version() == CURRENT_SCHEMA_VERSION
            # Verify paper_enrichments table exists
            tables = store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = [t["name"] for t in tables]
            assert "paper_enrichments" in table_names
            assert "paper_exports" in table_names


class TestEnrichments:
    def test_save_and_get_enrichment(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))
            store.save_enrichment(
                {
                    "arxiv_id": "2401.00001v1",
                    "pwc_id": "test-paper",
                    "has_code": True,
                    "code_urls": ["https://github.com/foo/bar"],
                    "datasets": ["ImageNet"],
                    "tasks": ["Image Classification"],
                }
            )
            enrichments = store.get_enrichments(["2401.00001v1"])
            assert len(enrichments) == 1
            e = enrichments["2401.00001v1"]
            assert e["has_code"] is True
            assert e["code_urls"] == ["https://github.com/foo/bar"]
            assert e["datasets"] == ["ImageNet"]
            assert e["tasks"] == ["Image Classification"]

    def test_save_enrichments_batch(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))
            store.upsert_paper(_make_paper(arxiv_id="2401.00002v1"))
            store.save_enrichments(
                {
                    "2401.00001v1": {
                        "arxiv_id": "2401.00001v1",
                        "pwc_id": "p1",
                        "has_code": True,
                        "code_urls": [],
                        "datasets": [],
                        "tasks": [],
                    },
                    "2401.00002v1": {
                        "arxiv_id": "2401.00002v1",
                        "pwc_id": "p2",
                        "has_code": False,
                        "code_urls": [],
                        "datasets": ["CIFAR-10"],
                        "tasks": [],
                    },
                }
            )
            enrichments = store.get_enrichments(["2401.00001v1", "2401.00002v1"])
            assert len(enrichments) == 2

    def test_get_enrichments_empty(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            result = store.get_enrichments([])
            assert result == {}

    def test_scores_include_enrichment_data(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))
            store.save_enrichment(
                {
                    "arxiv_id": "2401.00001v1",
                    "pwc_id": "test-paper",
                    "has_code": True,
                    "code_urls": ["https://github.com/foo/bar"],
                    "datasets": ["ImageNet"],
                    "tasks": ["Image Classification"],
                }
            )
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(
                run_id,
                [
                    {
                        "arxiv_id": "2401.00001v1",
                        "score_total": 0.8,
                        "keyword_score": 0.5,
                        "category_score": 0.2,
                        "recency_score": 0.1,
                    }
                ],
            )
            scores = store.get_scores_for_run(run_id)
            assert scores[0]["has_code"] is True
            assert scores[0]["datasets"] == ["ImageNet"]
            assert scores[0]["tasks"] == ["Image Classification"]

    def test_scores_without_enrichment(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper(arxiv_id="2401.00001v1"))
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(
                run_id,
                [
                    {
                        "arxiv_id": "2401.00001v1",
                        "score_total": 0.8,
                        "keyword_score": 0.5,
                        "category_score": 0.2,
                        "recency_score": 0.1,
                    }
                ],
            )
            scores = store.get_scores_for_run(run_id)
            assert scores[0]["has_code"] is False
            assert scores[0]["datasets"] == []
            assert scores[0]["tasks"] == []


class TestPaperExports:
    def test_record_and_get_exported_ids(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.record_export(
                "2401.00001v1", "github_issue", "https://github.com/foo/bar/issues/1"
            )
            exported = store.get_exported_ids("github_issue")
            assert "2401.00001v1" in exported

    def test_dedup_exports(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.record_export("2401.00001v1", "github_issue", "url1")
            store.record_export("2401.00001v1", "github_issue", "url2")
            exported = store.get_exported_ids("github_issue")
            assert len(exported) == 1

    def test_different_export_types(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.record_export("2401.00001v1", "github_issue", "url1")
            store.record_export("2401.00001v1", "slack", "url2")
            gh_exported = store.get_exported_ids("github_issue")
            slack_exported = store.get_exported_ids("slack")
            assert len(gh_exported) == 1
            assert len(slack_exported) == 1

    def test_empty_exports(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            exported = store.get_exported_ids("github_issue")
            assert exported == set()


class TestSchemaMigrationV2:
    def test_v1_db_gets_migrated(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = tmp_path / "v1.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""\
            CREATE TABLE papers (
                arxiv_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                authors TEXT NOT NULL, abstract TEXT NOT NULL,
                categories TEXT NOT NULL, published TEXT NOT NULL,
                updated TEXT, url TEXT NOT NULL, pdf_url TEXT,
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
            );
            CREATE TABLE runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time TEXT NOT NULL, queries_used TEXT NOT NULL,
                papers_new INTEGER NOT NULL, papers_seen INTEGER NOT NULL
            );
            CREATE TABLE paper_scores (
                arxiv_id TEXT NOT NULL, run_id INTEGER NOT NULL,
                score_total REAL NOT NULL, keyword_score REAL,
                category_score REAL, recency_score REAL,
                matched_query TEXT, PRIMARY KEY (arxiv_id, run_id)
            );
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (1);
        """)
        conn.close()

        with PaperStore(db_path) as store:
            assert store.schema_version() == CURRENT_SCHEMA_VERSION
            # Verify the new columns exist by inserting data with them
            store.upsert_paper(_make_paper())
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(
                run_id,
                [
                    {
                        "arxiv_id": "2401.12345v1",
                        "score_total": 0.8,
                        "keyword_score": 0.5,
                        "category_score": 0.2,
                        "recency_score": 0.1,
                        "embedding_score": 0.75,
                        "citation_score": 0.3,
                    },
                ],
            )
            scores = store.get_scores_for_run(run_id)
            assert scores[0]["embedding_score"] == 0.75
            assert scores[0]["citation_score"] == 0.3

    def test_save_scores_with_embedding_and_citation(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_make_paper())
            run_id = store.record_run(["q1"], 1, 0)
            store.save_scores(
                run_id,
                [
                    {
                        "arxiv_id": "2401.12345v1",
                        "score_total": 0.9,
                        "keyword_score": 0.6,
                        "category_score": 0.3,
                        "recency_score": 0.1,
                        "embedding_score": 0.85,
                        "citation_score": None,
                    },
                ],
            )
            scores = store.get_scores_for_run(run_id)
            assert len(scores) == 1
            assert scores[0]["embedding_score"] == 0.85
            assert scores[0]["citation_score"] is None
