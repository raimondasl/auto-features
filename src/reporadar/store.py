"""SQLite storage for papers, runs, and scores."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CURRENT_SCHEMA_VERSION = 9

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
    embedding_score REAL,
    citation_score  REAL,
    rrf_score       REAL,
    matched_query   TEXT,
    PRIMARY KEY (arxiv_id, run_id)
);

CREATE TABLE IF NOT EXISTS paper_llm_scores (
    arxiv_id    TEXT NOT NULL,
    run_id      INTEGER NOT NULL,
    llm_score   INTEGER NOT NULL,
    llm_reason  TEXT,
    PRIMARY KEY (arxiv_id, run_id)
);

CREATE TABLE IF NOT EXISTS paper_enrichments (
    arxiv_id    TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
    pwc_id      TEXT,
    has_code    INTEGER NOT NULL DEFAULT 0,
    code_urls   TEXT,
    datasets    TEXT,
    tasks       TEXT,
    models      TEXT,
    upvotes     INTEGER NOT NULL DEFAULT 0,
    fetched_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_exports (
    arxiv_id    TEXT NOT NULL,
    export_type TEXT NOT NULL,
    export_ref  TEXT,
    exported_at TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, export_type)
);

CREATE TABLE IF NOT EXISTS workspace_repos (
    repo_id     TEXT PRIMARY KEY,
    repo_path   TEXT NOT NULL,
    config_path TEXT,
    added_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repo_paper_scores (
    repo_id         TEXT NOT NULL REFERENCES workspace_repos(repo_id),
    arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id),
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    score_total     REAL NOT NULL,
    keyword_score   REAL,
    category_score  REAL,
    recency_score   REAL,
    embedding_score REAL,
    citation_score  REAL,
    matched_query   TEXT,
    PRIMARY KEY (repo_id, arxiv_id, run_id)
);

CREATE TABLE IF NOT EXISTS keyword_frequencies (
    run_id      INTEGER NOT NULL,
    keyword     TEXT NOT NULL,
    frequency   INTEGER NOT NULL,
    PRIMARY KEY (run_id, keyword)
);

CREATE TABLE IF NOT EXISTS paper_stars (
    arxiv_id    TEXT PRIMARY KEY,
    starred_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_ratings (
    arxiv_id    TEXT PRIMARY KEY,
    rating      INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    rated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
    arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id),
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (arxiv_id, model)
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""

# Maps target_version -> list of SQL statements to upgrade from previous version.
MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE paper_scores ADD COLUMN embedding_score REAL",
        "ALTER TABLE paper_scores ADD COLUMN citation_score REAL",
    ],
    3: [
        """\
        CREATE TABLE IF NOT EXISTS paper_enrichments (
            arxiv_id    TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
            pwc_id      TEXT,
            has_code    INTEGER NOT NULL DEFAULT 0,
            code_urls   TEXT,
            datasets    TEXT,
            tasks       TEXT,
            fetched_at  TEXT NOT NULL
        )""",
        """\
        CREATE TABLE IF NOT EXISTS paper_exports (
            arxiv_id    TEXT NOT NULL,
            export_type TEXT NOT NULL,
            export_ref  TEXT,
            exported_at TEXT NOT NULL,
            PRIMARY KEY (arxiv_id, export_type)
        )""",
    ],
    4: [
        """\
        CREATE TABLE IF NOT EXISTS workspace_repos (
            repo_id     TEXT PRIMARY KEY,
            repo_path   TEXT NOT NULL,
            config_path TEXT,
            added_at    TEXT NOT NULL
        )""",
        """\
        CREATE TABLE IF NOT EXISTS repo_paper_scores (
            repo_id         TEXT NOT NULL REFERENCES workspace_repos(repo_id),
            arxiv_id        TEXT NOT NULL REFERENCES papers(arxiv_id),
            run_id          INTEGER NOT NULL REFERENCES runs(run_id),
            score_total     REAL NOT NULL,
            keyword_score   REAL,
            category_score  REAL,
            recency_score   REAL,
            embedding_score REAL,
            citation_score  REAL,
            matched_query   TEXT,
            PRIMARY KEY (repo_id, arxiv_id, run_id)
        )""",
    ],
    5: [
        """\
        CREATE TABLE IF NOT EXISTS keyword_frequencies (
            run_id      INTEGER NOT NULL,
            keyword     TEXT NOT NULL,
            frequency   INTEGER NOT NULL,
            PRIMARY KEY (run_id, keyword)
        )""",
        """\
        CREATE TABLE IF NOT EXISTS paper_stars (
            arxiv_id    TEXT PRIMARY KEY,
            starred_at  TEXT NOT NULL
        )""",
        """\
        CREATE TABLE IF NOT EXISTS paper_ratings (
            arxiv_id    TEXT PRIMARY KEY,
            rating      INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
            rated_at    TEXT NOT NULL
        )""",
    ],
    6: [
        # Hugging Face Papers enrichment adds linked models and community
        # upvotes alongside the existing code/dataset links.
        "ALTER TABLE paper_enrichments ADD COLUMN models TEXT",
        "ALTER TABLE paper_enrichments ADD COLUMN upvotes INTEGER NOT NULL DEFAULT 0",
    ],
    7: [
        # Feature 6: per-run LLM actionability scores for repo-aware triage.
        """\
        CREATE TABLE IF NOT EXISTS paper_llm_scores (
            arxiv_id    TEXT NOT NULL,
            run_id      INTEGER NOT NULL,
            llm_score   INTEGER NOT NULL,
            llm_reason  TEXT,
            PRIMARY KEY (arxiv_id, run_id)
        )""",
    ],
    8: [
        # Hybrid retrieval (roadmap #4): RRF-fused rank of the heuristic + BM25
        # lexical orderings. When set, the digest orders by it instead of score_total.
        "ALTER TABLE paper_scores ADD COLUMN rrf_score REAL",
    ],
    9: [
        # Persistent per-paper embedding cache: compute a paper's vector once and
        # reuse it across runs (instead of re-encoding every paper every run).
        # A plain BLOB table — no native extension needed to read the cache; the
        # optional sqlite-vec index (reporadar.vec_index) is built lazily on top.
        """\
        CREATE TABLE IF NOT EXISTS paper_embeddings (
            arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id),
            model       TEXT NOT NULL,
            dim         INTEGER NOT NULL,
            vec         BLOB NOT NULL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (arxiv_id, model)
        )
        """,
    ],
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StoreError(Exception):
    """Raised when the database cannot be opened or is corrupt."""


class PaperStore:
    """Manages the SQLite database for storing arXiv papers and run metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.db_path), timeout=5)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA quick_check")
            self._init_schema()
        except sqlite3.DatabaseError as exc:
            raise StoreError(
                f"Cannot open database {self.db_path}: {exc}. "
                "The file may be corrupt — try deleting it and running `rr update` again."
            ) from exc
        except sqlite3.OperationalError as exc:
            raise StoreError(
                f"Cannot open database {self.db_path}: {exc}. "
                "The database may be locked by another process."
            ) from exc

    def _init_schema(self) -> None:
        # Check if this is a fresh DB or existing one
        has_schema_version = (
            self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            ).fetchone()
            is not None
        )

        has_tables = (
            self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='papers'"
            ).fetchone()
            is not None
        )

        if not has_tables:
            # Fresh DB — create all tables including schema_version
            self._conn.executescript(SCHEMA_SQL)
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
            self._conn.commit()
            return

        if not has_schema_version:
            # Legacy DB — has tables but no schema_version. Bootstrap at version 1.
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (CURRENT_SCHEMA_VERSION,),
            )
            self._conn.commit()
            return

        # Existing DB with schema_version — check if migration needed
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        current = row["version"] if row else 0

        if current < CURRENT_SCHEMA_VERSION:
            for target in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
                if target in MIGRATIONS:
                    for sql in MIGRATIONS[target]:
                        self._conn.execute(sql)
            self._conn.execute(
                "UPDATE schema_version SET version = ?",
                (CURRENT_SCHEMA_VERSION,),
            )
            self._conn.commit()

    def schema_version(self) -> int:
        """Return the current schema version."""
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        return row["version"] if row else 0

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
        row = self._conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
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
        return int(row[0])

    def save_paper_embedding(self, arxiv_id: str, model: str, dim: int, vec_bytes: bytes) -> None:
        """Cache a paper's embedding vector (raw little-endian float32 bytes)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO paper_embeddings (arxiv_id, model, dim, vec, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (arxiv_id, model, dim, vec_bytes, _now_iso()),
        )
        self._conn.commit()

    def get_embedding_blobs(
        self, model: str, arxiv_ids: list[str] | None = None
    ) -> dict[str, bytes]:
        """Return cached embedding blobs ``{arxiv_id: vec_bytes}`` for *model*.

        With *arxiv_ids*, restrict to those ids; otherwise return the whole cache
        for the model. Callers that would exceed SQLite's parameter limit should
        pass ``None`` and filter in Python.
        """
        if arxiv_ids is None:
            rows = self._conn.execute(
                "SELECT arxiv_id, vec FROM paper_embeddings WHERE model = ?", (model,)
            ).fetchall()
        else:
            ids = list(arxiv_ids)
            if not ids:
                return {}
            placeholders = ",".join("?" * len(ids))
            rows = self._conn.execute(
                "SELECT arxiv_id, vec FROM paper_embeddings "
                f"WHERE model = ? AND arxiv_id IN ({placeholders})",
                (model, *ids),
            ).fetchall()
        return {r["arxiv_id"]: r["vec"] for r in rows}

    def save_paper_embeddings(self, rows: list[tuple[str, str, int, bytes]]) -> None:
        """Batch-cache embeddings ``(arxiv_id, model, dim, vec_bytes)`` in one commit."""
        if not rows:
            return
        now = _now_iso()
        self._conn.executemany(
            "INSERT OR REPLACE INTO paper_embeddings (arxiv_id, model, dim, vec, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(arxiv_id, model, dim, vec, now) for (arxiv_id, model, dim, vec) in rows],
        )
        self._conn.commit()

    def clear_embeddings(self, model: str | None = None) -> int:
        """Delete cached embeddings (all, or one *model*). Returns the row count deleted."""
        if model is None:
            cur = self._conn.execute("DELETE FROM paper_embeddings")
        else:
            cur = self._conn.execute("DELETE FROM paper_embeddings WHERE model = ?", (model,))
        self._conn.commit()
        return int(cur.rowcount)

    def embedding_count(self, model: str | None = None) -> int:
        """Number of cached embeddings (optionally for a single model)."""
        if model is None:
            row = self._conn.execute("SELECT COUNT(*) FROM paper_embeddings").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM paper_embeddings WHERE model = ?", (model,)
            ).fetchone()
        return int(row[0])

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

    def get_runs(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return all runs ordered by time descending.

        If *limit* is given, return at most that many runs.
        """
        # Tiebreak by run_id so runs recorded within the same clock tick
        # (identical run_time) still order deterministically newest-first.
        sql = "SELECT * FROM runs ORDER BY run_time DESC, run_id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._conn.execute(sql).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["queries_used"] = json.loads(d["queries_used"])
            result.append(d)
        return result

    def get_last_run(self) -> dict[str, Any] | None:
        """Return the most recent run, or None."""
        row = self._conn.execute(
            "SELECT * FROM runs ORDER BY run_time DESC, run_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["queries_used"] = json.loads(d["queries_used"])
        return d

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """Return the run with *run_id* (queries_used decoded), or None."""
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["queries_used"] = json.loads(d["queries_used"])
        return d

    def get_previous_run_id(self, run_id: int) -> int | None:
        """Find the run immediately before *run_id*, or None."""
        row = self._conn.execute(
            "SELECT run_id FROM runs WHERE run_id < ? ORDER BY run_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return row["run_id"] if row else None

    def get_scored_paper_ids_for_run(self, run_id: int) -> set[str]:
        """Return the set of arxiv_ids that were scored in *run_id*."""
        rows = self._conn.execute(
            "SELECT arxiv_id FROM paper_scores WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {row["arxiv_id"] for row in rows}

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
                        embedding_score, citation_score, rrf_score,
                        matched_query)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    s["arxiv_id"],
                    run_id,
                    s["score_total"],
                    s.get("keyword_score"),
                    s.get("category_score"),
                    s.get("recency_score"),
                    s.get("embedding_score"),
                    s.get("citation_score"),
                    s.get("rrf_score"),
                    s.get("matched_query"),
                ),
            )
        self._conn.commit()

    def get_scores_for_run(self, run_id: int) -> list[dict[str, Any]]:
        """Return scores for a given run, ordered by score descending."""
        rows = self._conn.execute(
            """\
            SELECT ps.*, p.title, p.url, p.abstract, p.authors, p.categories, p.published,
                   pe.has_code, pe.code_urls AS enrichment_code_urls,
                   pe.datasets AS enrichment_datasets, pe.tasks AS enrichment_tasks,
                   pe.models AS enrichment_models, pe.upvotes AS enrichment_upvotes,
                   pls.llm_score, pls.llm_reason
              FROM paper_scores ps
              JOIN papers p ON ps.arxiv_id = p.arxiv_id
              LEFT JOIN paper_enrichments pe ON ps.arxiv_id = pe.arxiv_id
              LEFT JOIN paper_llm_scores pls
                     ON ps.arxiv_id = pls.arxiv_id AND ps.run_id = pls.run_id
             WHERE ps.run_id = ?
             ORDER BY COALESCE(ps.rrf_score, ps.score_total) DESC""",
            (run_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["authors"] = json.loads(d["authors"])
            d["categories"] = json.loads(d["categories"])
            # Unpack enrichment fields
            d["has_code"] = bool(d.get("has_code"))
            raw_code_urls = d.pop("enrichment_code_urls", None)
            d["code_urls"] = json.loads(raw_code_urls) if raw_code_urls else []
            raw_datasets = d.pop("enrichment_datasets", None)
            d["datasets"] = json.loads(raw_datasets) if raw_datasets else []
            raw_tasks = d.pop("enrichment_tasks", None)
            d["tasks"] = json.loads(raw_tasks) if raw_tasks else []
            raw_models = d.pop("enrichment_models", None)
            d["models"] = json.loads(raw_models) if raw_models else []
            d["upvotes"] = d.get("enrichment_upvotes") or 0
            d.pop("enrichment_upvotes", None)
            # llm_score/llm_reason are left as-is (None when the paper wasn't triaged)
            result.append(d)
        return result

    def save_llm_scores(self, run_id: int, llm_scores: dict[str, dict[str, Any]]) -> None:
        """Persist LLM triage scores for a run: ``{arxiv_id: {llm_score, llm_reason}}``."""
        for arxiv_id, v in llm_scores.items():
            self._conn.execute(
                """\
                INSERT OR REPLACE INTO paper_llm_scores (arxiv_id, run_id, llm_score, llm_reason)
                VALUES (?, ?, ?, ?)""",
                (arxiv_id, run_id, int(v["llm_score"]), v.get("llm_reason", "")),
            )
        self._conn.commit()

    # ── Enrichment operations ─────────────────────────────────────────

    def save_enrichment(self, enrichment: dict[str, Any]) -> None:
        """Save a single paper enrichment (Hugging Face Papers or legacy PwC)."""
        self._conn.execute(
            """\
            INSERT OR REPLACE INTO paper_enrichments
                   (arxiv_id, pwc_id, has_code, code_urls, datasets, tasks,
                    models, upvotes, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                enrichment["arxiv_id"],
                enrichment.get("pwc_id") or enrichment.get("hf_id"),
                1 if enrichment.get("has_code") else 0,
                json.dumps(enrichment.get("code_urls", [])),
                json.dumps(enrichment.get("datasets", [])),
                json.dumps(enrichment.get("tasks", [])),
                json.dumps(enrichment.get("models", [])),
                int(enrichment.get("upvotes") or 0),
                _now_iso(),
            ),
        )
        self._conn.commit()

    def save_enrichments(self, enrichments: dict[str, dict[str, Any]]) -> None:
        """Save multiple paper enrichments."""
        for enrichment in enrichments.values():
            self._conn.execute(
                """\
                INSERT OR REPLACE INTO paper_enrichments
                       (arxiv_id, pwc_id, has_code, code_urls, datasets, tasks,
                        models, upvotes, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    enrichment["arxiv_id"],
                    enrichment.get("pwc_id") or enrichment.get("hf_id"),
                    1 if enrichment.get("has_code") else 0,
                    json.dumps(enrichment.get("code_urls", [])),
                    json.dumps(enrichment.get("datasets", [])),
                    json.dumps(enrichment.get("tasks", [])),
                    json.dumps(enrichment.get("models", [])),
                    int(enrichment.get("upvotes") or 0),
                    _now_iso(),
                ),
            )
        self._conn.commit()

    def get_enrichments(self, arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get enrichments for a list of arXiv IDs."""
        if not arxiv_ids:
            return {}
        placeholders = ",".join("?" for _ in arxiv_ids)
        rows = self._conn.execute(
            f"SELECT * FROM paper_enrichments WHERE arxiv_id IN ({placeholders})",
            arxiv_ids,
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            d = dict(row)
            d["has_code"] = bool(d["has_code"])
            d["code_urls"] = json.loads(d["code_urls"]) if d["code_urls"] else []
            d["datasets"] = json.loads(d["datasets"]) if d["datasets"] else []
            d["tasks"] = json.loads(d["tasks"]) if d["tasks"] else []
            d["models"] = json.loads(d["models"]) if d.get("models") else []
            d["upvotes"] = d.get("upvotes") or 0
            result[d["arxiv_id"]] = d
        return result

    # ── Export tracking ───────────────────────────────────────────────

    def record_export(self, arxiv_id: str, export_type: str, export_ref: str | None) -> None:
        """Record that a paper was exported (e.g. as a GitHub issue)."""
        self._conn.execute(
            """\
            INSERT OR REPLACE INTO paper_exports
                   (arxiv_id, export_type, export_ref, exported_at)
            VALUES (?, ?, ?, ?)""",
            (arxiv_id, export_type, export_ref, _now_iso()),
        )
        self._conn.commit()

    def get_exported_ids(self, export_type: str) -> set[str]:
        """Return the set of arxiv_ids that have been exported as *export_type*."""
        rows = self._conn.execute(
            "SELECT arxiv_id FROM paper_exports WHERE export_type = ?",
            (export_type,),
        ).fetchall()
        return {row["arxiv_id"] for row in rows}

    # ── Workspace operations ──────────────────────────────────────────

    def add_workspace_repo(
        self, repo_id: str, repo_path: str, config_path: str | None = None
    ) -> None:
        """Register a repo in the workspace."""
        self._conn.execute(
            """\
            INSERT OR REPLACE INTO workspace_repos
                   (repo_id, repo_path, config_path, added_at)
            VALUES (?, ?, ?, ?)""",
            (repo_id, repo_path, config_path, _now_iso()),
        )
        self._conn.commit()

    def remove_workspace_repo(self, repo_id: str) -> bool:
        """Unregister a repo. Returns True if it existed."""
        cur = self._conn.execute(
            "DELETE FROM workspace_repos WHERE repo_id = ?",
            (repo_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_workspace_repos(self) -> list[dict[str, Any]]:
        """Return all registered workspace repos."""
        rows = self._conn.execute("SELECT * FROM workspace_repos ORDER BY added_at").fetchall()
        return [dict(r) for r in rows]

    def save_repo_scores(
        self,
        repo_id: str,
        run_id: int,
        scores: list[dict[str, Any]],
    ) -> None:
        """Save per-repo paper scores for a run."""
        for s in scores:
            self._conn.execute(
                """\
                INSERT OR REPLACE INTO repo_paper_scores
                       (repo_id, arxiv_id, run_id, score_total,
                        keyword_score, category_score, recency_score,
                        embedding_score, citation_score, matched_query)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    repo_id,
                    s["arxiv_id"],
                    run_id,
                    s["score_total"],
                    s.get("keyword_score"),
                    s.get("category_score"),
                    s.get("recency_score"),
                    s.get("embedding_score"),
                    s.get("citation_score"),
                    s.get("matched_query"),
                ),
            )
        self._conn.commit()

    def get_repo_scores_for_run(
        self, run_id: int, repo_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return per-repo scores for a run, optionally filtered by repo_id."""
        if repo_id:
            rows = self._conn.execute(
                """\
                SELECT rps.*, p.title, p.url, p.abstract, p.authors, p.categories, p.published
                  FROM repo_paper_scores rps
                  JOIN papers p ON rps.arxiv_id = p.arxiv_id
                 WHERE rps.run_id = ? AND rps.repo_id = ?
                 ORDER BY rps.score_total DESC""",
                (run_id, repo_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """\
                SELECT rps.*, p.title, p.url, p.abstract, p.authors, p.categories, p.published
                  FROM repo_paper_scores rps
                  JOIN papers p ON rps.arxiv_id = p.arxiv_id
                 WHERE rps.run_id = ?
                 ORDER BY rps.score_total DESC""",
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["authors"] = json.loads(d["authors"])
            d["categories"] = json.loads(d["categories"])
            result.append(d)
        return result

    # ── Keyword frequency operations ─────────────────────────────────

    def save_keyword_frequencies(self, run_id: int, frequencies: dict[str, int]) -> None:
        """Save keyword frequency counts for a run."""
        for keyword, freq in frequencies.items():
            self._conn.execute(
                """\
                INSERT OR REPLACE INTO keyword_frequencies (run_id, keyword, frequency)
                VALUES (?, ?, ?)""",
                (run_id, keyword, freq),
            )
        self._conn.commit()

    def get_keyword_frequencies(self, run_id: int) -> dict[str, int]:
        """Return keyword frequencies for a given run."""
        rows = self._conn.execute(
            "SELECT keyword, frequency FROM keyword_frequencies WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {row["keyword"]: row["frequency"] for row in rows}

    # ── Star operations ──────────────────────────────────────────────

    def star_paper(self, arxiv_id: str) -> None:
        """Star a paper."""
        self._conn.execute(
            "INSERT OR REPLACE INTO paper_stars (arxiv_id, starred_at) VALUES (?, ?)",
            (arxiv_id, _now_iso()),
        )
        self._conn.commit()

    def get_starred_papers(self) -> list[str]:
        """Return list of starred arxiv_ids."""
        rows = self._conn.execute(
            "SELECT arxiv_id FROM paper_stars ORDER BY starred_at DESC"
        ).fetchall()
        return [row["arxiv_id"] for row in rows]

    # ── Rating operations ────────────────────────────────────────────

    def save_rating(self, arxiv_id: str, rating: int) -> None:
        """Save a user rating (1-5) for a paper."""
        self._conn.execute(
            """\
            INSERT OR REPLACE INTO paper_ratings (arxiv_id, rating, rated_at)
            VALUES (?, ?, ?)""",
            (arxiv_id, rating, _now_iso()),
        )
        self._conn.commit()

    def get_rating(self, arxiv_id: str) -> int | None:
        """Return the rating for a paper, or None if not rated."""
        row = self._conn.execute(
            "SELECT rating FROM paper_ratings WHERE arxiv_id = ?",
            (arxiv_id,),
        ).fetchone()
        return row["rating"] if row else None

    def get_all_ratings(self) -> dict[str, int]:
        """Return all ratings as {arxiv_id: rating}."""
        rows = self._conn.execute("SELECT arxiv_id, rating FROM paper_ratings").fetchall()
        return {row["arxiv_id"]: row["rating"] for row in rows}

    def get_rated_paper_scores(self) -> list[dict[str, Any]]:
        """Return score dicts for all rated papers (from their most recent run)."""
        rows = self._conn.execute(
            """\
            SELECT pr.arxiv_id, pr.rating,
                   ps.score_total, ps.keyword_score, ps.category_score,
                   ps.recency_score, ps.embedding_score, ps.citation_score
              FROM paper_ratings pr
              JOIN paper_scores ps ON pr.arxiv_id = ps.arxiv_id
             WHERE ps.run_id = (
                SELECT MAX(run_id) FROM paper_scores WHERE arxiv_id = pr.arxiv_id
             )"""
        ).fetchall()
        return [dict(row) for row in rows]
