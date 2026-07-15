"""Tests for reporadar.archive — the GitHub Pages digest archive."""

from __future__ import annotations

import json
from pathlib import Path

from reporadar.archive import INDEX_NAME, MANIFEST_NAME, archive_digest
from reporadar.store import PaperStore


def _paper(arxiv_id: str, title: str = "A Paper") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["Alice"],
        "abstract": "We propose a concrete method with a benchmark.",
        "categories": ["cs.LG"],
        "published": "2024-01-01T00:00:00+00:00",
        "updated": "2024-01-01T00:00:00+00:00",
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
    }


def _seed(store: PaperStore) -> int:
    store.upsert_paper(_paper("2401.00001v1", "First LoRA Paper"))
    store.upsert_paper(_paper("2401.00002v1", "Second Paper"))
    run_id = store.record_run(["q1"], papers_new=2, papers_seen=0)
    store.save_scores(
        run_id,
        [
            {
                "arxiv_id": "2401.00001v1",
                "score_total": 0.9,
                "keyword_score": 0.6,
                "category_score": 0.3,
                "recency_score": 0.2,
            },
            {
                "arxiv_id": "2401.00002v1",
                "score_total": 0.4,
                "keyword_score": 0.2,
                "category_score": 0.1,
                "recency_score": 0.1,
            },
        ],
    )
    return run_id


class TestArchiveDigest:
    def test_writes_dated_entry_index_and_manifest(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)
            adir = tmp_path / "site"
            entry, index = archive_digest(store, run_id, adir, date_str="2026-07-15")

        assert entry.name == "2026-07-15.html"
        assert entry.exists()
        assert index.name == INDEX_NAME and index.exists()
        assert (adir / MANIFEST_NAME).exists()
        assert "<!DOCTYPE html>" in entry.read_text(encoding="utf-8")
        # The index links to the dated edition.
        assert "2026-07-15.html" in index.read_text(encoding="utf-8")

    def test_manifest_dedups_by_date(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)
            adir = tmp_path / "site"
            archive_digest(store, run_id, adir, date_str="2026-07-15")
            archive_digest(store, run_id, adir, date_str="2026-07-15")  # same date

        manifest = json.loads((adir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert len(manifest) == 1

    def test_multiple_dates_sorted_newest_first(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)
            adir = tmp_path / "site"
            archive_digest(store, run_id, adir, date_str="2026-07-14")
            archive_digest(store, run_id, adir, date_str="2026-07-16")
            archive_digest(store, run_id, adir, date_str="2026-07-15")

        manifest = json.loads((adir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert [e["date"] for e in manifest] == ["2026-07-16", "2026-07-15", "2026-07-14"]
        index_html = (adir / INDEX_NAME).read_text(encoding="utf-8")
        # Newest edition appears before the oldest in the rendered index.
        assert index_html.index("2026-07-16") < index_html.index("2026-07-14")

    def test_survives_corrupt_manifest(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)
            adir = tmp_path / "site"
            adir.mkdir()
            (adir / MANIFEST_NAME).write_text("{ this is not valid json", encoding="utf-8")

            entry, index = archive_digest(store, run_id, adir, date_str="2026-07-15")

        assert entry.exists() and index.exists()
        manifest = json.loads((adir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert len(manifest) == 1 and manifest[0]["date"] == "2026-07-15"

    def test_defaults_date_to_today_and_counts_picks(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)
            adir = tmp_path / "site"
            entry, _ = archive_digest(store, run_id, adir)

        assert entry.suffix == ".html" and entry.exists()
        manifest = json.loads((adir / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert len(manifest) == 1
        # The 0.9-scored paper lands in Top Picks; the 0.4 one does not.
        assert manifest[0]["top_picks"] == 1
        assert manifest[0]["total_scored"] == 2
