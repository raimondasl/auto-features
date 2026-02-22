"""SQLite storage for papers, runs, and scores."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    authors     TEXT NOT NULL,
    abstract    TEXT NOT NULL,
    categories  TEXT NOT NULL,
    published   TEXT NOT NULL,
    updated     TEXT,
    url         TEXT NOT NULL,
    pdf_url     TEXT,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_time        TEXT NOT NULL,
    queries_used    TEXT NOT NULL,
    papers_new      INTEGER NOT NULL,
    papers_seen     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_scores (
    arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id),
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    score_total     REAL NOT NULL,
    keyword_score   REAL,
    category_score  REAL,
    recency_score   REAL,
    matched_query   TEXT,
    PRIMARY KEY (arxiv_id, run_id)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperStore:
    """Manages the SQLite database for storing arXiv papers and run metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PaperStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Paper operations ───────────────────────────────────────────────

    def upsert_paper(self, paper: dict[str, Any]) -> bool:
        """Insert or update a paper. Returns True if the paper was new."""
        now = _now_iso()
        existing = self._conn.execute(
            "SELECT arxiv_id FROM papers WHERE arxiv_id = ?",
            (paper["arxiv_id"],),
        ).fetchone()

        if existing:
            self._conn.execute(
                """\
                UPDATE papers
                   SET title = ?, authors = ?, abstract = ?, categories = ?,
                       published = ?, updated = ?, url = ?, pdf_url = ?,
                       last_seen = ?
                 WHERE arxiv_id = ?""",
                (
                    paper["title"],
                    json.dumps(paper["authors"]),
                    paper["abstract"],
                    json.dumps(paper["categories"]),
                    paper["published"],
                    paper.get("updated"),
                    paper["url"],
                    paper.get("pdf_url"),
                    now,
                    paper["arxiv_id"],
                ),
            )
            self._conn.commit()
            return False

        self._conn.execute(
            """\
            INSERT INTO papers
                   (arxiv_id, title, authors, abstract, categories,
                    published, updated, url, pdf_url, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper["arxiv_id"],
                paper["title"],
                json.dumps(paper["authors"]),
                paper["abstract"],
                json.dumps(paper["categories"]),
                paper["published"],
                paper.get("updated"),
                paper["url"],
                paper.get("pdf_url"),
                now,
                now,
            ),
        )
        self._conn.commit()
        return True

    def upsert_papers(self, papers: list[dict[str, Any]]) -> tuple[int, int]:
        """Insert/update a batch of papers. Returns (new_count, seen_count)."""
        new = 0
        seen = 0
        for paper in papers:
            if self.upsert_paper(paper):
                new += 1
            else:
                seen += 1
        self._conn.commit()
        return new, seen

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Fetch a single paper by its arXiv ID."""
        row = self._conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def get_all_papers(self) -> list[dict[str, Any]]:
        """Return all papers in the database."""
        rows = self._conn.execute("SELECT * FROM papers ORDER BY published DESC").fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_papers_since(self, since_iso: str) -> list[dict[str, Any]]:
        """Return papers first seen on or after *since_iso*."""
        rows = self._conn.execute(
            "SELECT * FROM papers WHERE first_seen >= ? ORDER BY published DESC",
            (since_iso,),
        ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def paper_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()
        return row[0]

    @staticmethod
    def _row_to_paper(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["authors"] = json.loads(d["authors"])
        d["categories"] = json.loads(d["categories"])
        return d

    # ── Run operations ─────────────────────────────────────────────────

    def record_run(
        self,
        queries_used: list[str],
        papers_new: int,
        papers_seen: int,
    ) -> int:
        """Record a collection run. Returns the run_id."""
        cur = self._conn.execute(
            """\
            INSERT INTO runs (run_time, queries_used, papers_new, papers_seen)
            VALUES (?, ?, ?, ?)""",
            (_now_iso(), json.dumps(queries_used), papers_new, papers_seen),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_runs(self) -> list[dict[str, Any]]:
        """Return all runs ordered by time descending."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY run_time DESC"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["queries_used"] = json.loads(d["queries_used"])
            result.append(d)
        return result

    def get_last_run(self) -> dict[str, Any] | None:
        """Return the most recent run, or None."""
        row = self._conn.execute(
            "SELECT * FROM runs ORDER BY run_time DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["queries_used"] = json.loads(d["queries_used"])
        return d

    # ── Score operations ───────────────────────────────────────────────

    def save_scores(
        self,
        run_id: int,
        scores: list[dict[str, Any]],
    ) -> None:
        """Save paper scores for a run.

        Each score dict should have: arxiv_id, score_total,
        keyword_score, category_score, recency_score, matched_query.
        """
        for s in scores:
            self._conn.execute(
                """\
                INSERT OR REPLACE INTO paper_scores
                       (arxiv_id, run_id, score_total,
                        keyword_score, category_score, recency_score,
                        matched_query)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    s["arxiv_id"],
                    run_id,
                    s["score_total"],
                    s.get("keyword_score"),
                    s.get("category_score"),
                    s.get("recency_score"),
                    s.get("matched_query"),
                ),
            )
        self._conn.commit()

    def get_scores_for_run(self, run_id: int) -> list[dict[str, Any]]:
        """Return scores for a given run, ordered by score descending."""
        rows = self._conn.execute(
            """\
            SELECT ps.*, p.title, p.url, p.abstract, p.authors, p.categories, p.published
              FROM paper_scores ps
              JOIN papers p ON ps.arxiv_id = p.arxiv_id
             WHERE ps.run_id = ?
             ORDER BY ps.score_total DESC""",
            (run_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["authors"] = json.loads(d["authors"])
            d["categories"] = json.loads(d["categories"])
            result.append(d)
        return result
