"""Pages archive — write dated HTML digests into a browsable static site.

``rr archive`` renders the latest run's digest to ``<archive-dir>/<date>.html``,
maintains a ``manifest.json`` (one entry per date, newest re-run wins), and
regenerates ``<archive-dir>/index.html`` listing every archived digest. Point
GitHub Pages at that directory (see the ``reporadar-action``) and each scheduled
run publishes a dated, linkable research radar.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reporadar.digest import (
    _load_html_template,
    categorize_papers,
    cited_ids_from,
    filter_since,
    generate_digest_html,
)
from reporadar.store import PaperStore

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.html"


def _load_manifest(archive_dir: Path) -> list[dict[str, Any]]:
    """Read the archive manifest, tolerating a missing or corrupt file."""
    path = archive_dir / MANIFEST_NAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _upsert_entry(manifest: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace any existing entry for the same date, then sort newest-first."""
    out = [e for e in manifest if e.get("date") != entry["date"]]
    out.append(entry)
    out.sort(key=lambda e: str(e.get("date", "")), reverse=True)
    return out


def render_index(entries: list[dict[str, Any]], generated_at: str) -> str:
    """Render the archive index page from manifest entries."""
    template = _load_html_template("archive_index.html.j2")
    rendered: str = template.render(entries=entries, generated_at=generated_at)
    return rendered


def archive_digest(
    store: PaperStore,
    run_id: int,
    archive_dir: str | Path,
    *,
    date_str: str | None = None,
    top_n: int = 15,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
) -> tuple[Path, Path]:
    """Write a dated HTML digest into *archive_dir* and rebuild its index.

    Returns ``(entry_path, index_path)``. *date_str* defaults to today's UTC date;
    re-running on the same date overwrites that day's entry (idempotent).
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")

    html = generate_digest_html(
        store,
        run_id,
        top_n=top_n,
        suggestions_config=suggestions_config,
        profile=profile,
        since_days=since_days,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
    )
    entry_file = f"{date_str}.html"
    entry_path = archive_dir / entry_file
    entry_path.write_text(html, encoding="utf-8")

    # Summary counts for the index card (mirror the digest's tiering).
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    top_picks, _, _ = categorize_papers(
        scored,
        top_n=top_n,
        cited_ids=cited_ids_from(profile),
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
    )
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    entry: dict[str, Any] = {
        "date": date_str,
        "file": entry_file,
        "run_id": run_id,
        "top_picks": len(top_picks),
        "total_scored": len(scored),
        "generated_at": generated_at,
    }
    manifest = _upsert_entry(_load_manifest(archive_dir), entry)
    (archive_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    index_path = archive_dir / INDEX_NAME
    index_path.write_text(render_index(manifest, generated_at), encoding="utf-8")

    return entry_path, index_path
