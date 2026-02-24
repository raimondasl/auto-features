"""Multi-repo workspace support with shared database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reporadar.store import PaperStore

WORKSPACE_DIR = Path.home() / ".reporadar"
WORKSPACE_DB = WORKSPACE_DIR / "workspace.db"


def ensure_workspace_dir() -> Path:
    """Create the workspace directory if it doesn't exist."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def open_workspace_store(db_path: Path | None = None) -> PaperStore:
    """Open (or create) the workspace database."""
    if db_path is None:
        ensure_workspace_dir()
        db_path = WORKSPACE_DB
    return PaperStore(db_path)


def score_papers_for_repo(
    repo_id: str,
    repo_path: str,
    papers: list[dict[str, Any]],
    cfg: Any,
) -> list[dict[str, Any]]:
    """Profile a single repo and rank papers against it.

    Returns a list of score dicts with ``repo_id`` attached.
    """
    from reporadar.profiler import profile_repo
    from reporadar.ranker import rank_papers

    profile = profile_repo(Path(repo_path))
    scores = rank_papers(
        papers,
        profile,
        cfg.ranking,
        cfg.queries,
        cfg.arxiv.categories,
        cfg.arxiv.lookback_days,
    )
    for s in scores:
        s["repo_id"] = repo_id
    return scores


def combined_digest_data(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Merge per-repo scores into a combined list.

    For each paper, the highest score across repos is used for sorting.
    Each paper gets a ``relevant_repos`` list of (repo_id, score) pairs.
    """
    all_scores = store.get_repo_scores_for_run(run_id)

    # Group by arxiv_id
    paper_map: dict[str, dict[str, Any]] = {}
    for s in all_scores:
        aid = s["arxiv_id"]
        if aid not in paper_map:
            paper_map[aid] = {
                "arxiv_id": aid,
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "abstract": s.get("abstract", ""),
                "authors": s.get("authors", []),
                "categories": s.get("categories", []),
                "published": s.get("published", ""),
                "max_score": s["score_total"],
                "relevant_repos": [],
            }
        entry = paper_map[aid]
        entry["relevant_repos"].append({"repo_id": s["repo_id"], "score": s["score_total"]})
        if s["score_total"] > entry["max_score"]:
            entry["max_score"] = s["score_total"]

    # Sort by max_score descending, take top_n
    result = sorted(paper_map.values(), key=lambda p: p["max_score"], reverse=True)
    return result[:top_n]
