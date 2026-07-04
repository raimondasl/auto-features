"""Shared logic for the RepoRadar eval benchmark.

Profiles a benchmark repo with the *real* RepoRadar profiler and ranks a
candidate pool with the *real* ranker, so the eval measures the shipping code
paths — not a reimplementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reporadar.config import ProfilerConfig, QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile, profile_repo
from reporadar.ranker import rank_papers

EVALS_DIR = Path(__file__).resolve().parent


def load_benchmark(path: str | Path | None = None) -> dict[str, Any]:
    """Load the benchmark case definitions from ``benchmark.yaml``."""
    path = Path(path) if path else EVALS_DIR / "benchmark.yaml"
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def resolve_repo_dir(case: dict[str, Any]) -> Path:
    """Absolute path to a case's offline mini-repo (``repo_dir`` in the YAML)."""
    return (EVALS_DIR / case["repo_dir"]).resolve()


def build_ranking_config(
    *,
    embeddings: bool = False,
    w_keyword: float = 1.0,
    w_category: float = 0.5,
    w_embedding: float = 1.5,
) -> RankingConfig:
    """Ranking weights for offline eval.

    Recency is fixed to 0 so results are deterministic regardless of when the
    benchmark runs (recency depends on wall-clock time). Embeddings are opt-in
    because they change with the installed model.
    """
    return RankingConfig(
        w_keyword=w_keyword,
        w_category=w_category,
        w_recency=0.0,
        w_embedding=w_embedding if embeddings else 0.0,
        w_citations=0.0,
    )


def profile_case_repo(repo_dir: Path, *, scan_source: bool = False) -> RepoProfile:
    """Profile a benchmark repo with the real profiler."""
    cfg = ProfilerConfig(scan_source=scan_source)
    return profile_repo(repo_dir, profiler_cfg=cfg)


def rank_pool(
    profile: RepoProfile,
    papers: list[dict[str, Any]],
    *,
    expected_categories: list[str],
    exclude: list[str] | None = None,
    ranking_cfg: RankingConfig | None = None,
    repo_dir: Path | None = None,
    embeddings: bool = False,
) -> list[dict[str, Any]]:
    """Rank a candidate pool with the real ranker; returns score dicts best-first.

    When *embeddings* is True and sentence-transformers is installed, the repo
    embedding is computed from *repo_dir* and used as an extra ranking signal.
    """
    ranking_cfg = ranking_cfg or build_ranking_config(embeddings=embeddings)
    queries_cfg = QueriesConfig(exclude=exclude or [])

    repo_embedding = None
    if embeddings and repo_dir is not None:
        try:
            from reporadar.embeddings import EMBEDDINGS_AVAILABLE, compute_repo_embedding

            if EMBEDDINGS_AVAILABLE:
                repo_embedding = compute_repo_embedding(repo_dir)
        except ImportError:
            repo_embedding = None

    return rank_papers(
        papers,
        profile,
        ranking_cfg,
        queries_cfg,
        expected_categories,
        lookback_days=3650,  # large; recency is zero-weighted anyway
        repo_embedding=repo_embedding,
    )
