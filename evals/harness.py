"""Shared logic for the RepoRadar eval benchmark.

Profiles a benchmark repo with the *real* RepoRadar profiler and ranks a
candidate pool with the *real* ranker, so the eval measures the shipping code
paths — not a reimplementation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from reporadar.config import ProfilerConfig, QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile, profile_repo
from reporadar.ranker import rank_papers

EVALS_DIR = Path(__file__).resolve().parent
WORK_DIR = EVALS_DIR / ".work"


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


# ── shared live helpers (used by both Tier A and Tier B live runs) ─────────


def clone_repo(url: str, dest: Path, *, reuse: bool = False) -> Path | None:
    """Shallow-clone *url* to *dest*. Reuse an existing clone if present."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"        ! clone failed: {exc}")
        return None
    return dest


def collect_live_papers(
    profile: RepoProfile,
    categories: list[str],
    *,
    sources: list[str],
    keys: dict[str, str],
    lookback_days: int,
    sort_by: str = "submitted",
) -> list[dict[str, Any]]:
    """Build queries and fetch papers from the requested live sources.

    ``sort_by="relevance"`` with a large ``lookback_days`` lets RepoRadar reach
    seminal older papers instead of only the recent fetch window.
    """
    from reporadar.collector import (
        CollectionError,
        build_queries,
        collect_papers,
        to_plain_keywords,
    )
    from reporadar.config import ArxivConfig

    arxiv_cfg = ArxivConfig(
        categories=categories or ["cs.LG", "cs.CL", "cs.CV", "cs.SE"],
        max_results_per_query=50,
        lookback_days=lookback_days,
        sort_by=sort_by,
    )
    queries = build_queries(profile, QueriesConfig(), arxiv_cfg)

    papers: list[dict[str, Any]] = []
    try:
        papers = collect_papers(queries, arxiv_cfg)
    except CollectionError as exc:
        # RAISE, do not degrade. This used to print a warning and return whatever it had —
        # which for a throttled case was nothing at all, and a 0-paper pool then scored as a
        # legitimate net@2 of 0.0 and entered the mean. `db` and `storage` were dropped that
        # way in the 2026-08-07 pool-300 arm and cost -17 of a -21 delta. A caller that
        # wants to continue past a failed case must say so by catching this.
        print(f"        ! arXiv collection FAILED (not an empty result): {exc}")
        raise

    seen = {p["arxiv_id"] for p in papers}
    # arXiv's boolean grammar is not a keyword query; every non-arXiv source below needs
    # the words out of it. See reporadar.collector.to_plain_keywords for what this
    # replaced and why the old one-liner silently stopped working.
    plain = [to_plain_keywords(q) for q in queries[:5]]

    if "openalex" in sources:
        from reporadar.sources.openalex import collect_papers as oa_collect

        for p in oa_collect(
            plain, lookback_days=lookback_days, api_key=keys.get("OPENALEX_API_KEY")
        ):
            if p["arxiv_id"] not in seen:
                papers.append(p)
                seen.add(p["arxiv_id"])

    if "semantic_scholar" in sources:
        from reporadar.sources.semantic_scholar import collect_papers as ss_collect

        for p in ss_collect(
            plain, api_key=keys.get("SEMANTIC_SCHOLAR_API_KEY"), lookback_days=lookback_days
        ):
            if p["arxiv_id"] not in seen:
                papers.append(p)
                seen.add(p["arxiv_id"])

    if "dblp" in sources:
        from reporadar.sources.dblp import collect_papers as dblp_collect

        for p in dblp_collect(plain, lookback_days=lookback_days):
            if p["arxiv_id"] not in seen:
                papers.append(p)
                seen.add(p["arxiv_id"])

    if "iacr" in sources:
        from reporadar.sources.iacr import collect_papers as iacr_collect

        for p in iacr_collect(plain, lookback_days=lookback_days):
            if p["arxiv_id"] not in seen:
                papers.append(p)
                seen.add(p["arxiv_id"])

    if "biorxiv" in sources:
        from reporadar.sources.biorxiv import collect_papers as bx_collect

        for p in bx_collect(plain, lookback_days=lookback_days):
            if p["arxiv_id"] not in seen:
                papers.append(p)
                seen.add(p["arxiv_id"])

    # Fail loudly on a source the harness can't fetch: silently ignoring it makes
    # a benchmark run look like a valid measurement of that source when it is a
    # no-op (exactly what happened the first time `--sources arxiv,dblp` was run).
    unknown = set(sources) - {"arxiv", "openalex", "semantic_scholar", "dblp", "biorxiv", "iacr"}
    if unknown:
        raise ValueError(
            f"Unknown eval source(s): {', '.join(sorted(unknown))}. "
            "Add a branch in collect_live_papers or fix --sources."
        )

    return papers


def assemble_repo_context(repo_dir: Path, *, max_readme: int = 3500) -> str:
    """A compact text summary of a repo for grounding the LLM judge/baseline.

    README excerpt + dependency manifests + a shallow file listing — enough to
    judge whether a paper's method is applicable to *this* codebase.
    """
    parts: list[str] = [f"Repository: {repo_dir.name}", ""]

    for readme in ("README.md", "README.rst", "README.txt", "readme.md"):
        p = repo_dir / readme
        if p.exists():
            parts += [
                "## README (excerpt)",
                p.read_text(encoding="utf-8", errors="ignore")[:max_readme],
                "",
            ]
            break

    for manifest in ("requirements.txt", "pyproject.toml", "package.json", "setup.py"):
        p = repo_dir / manifest
        if p.exists():
            parts += [f"## {manifest}", p.read_text(encoding="utf-8", errors="ignore")[:1200], ""]

    # Shallow listing of source files (names only) for a sense of structure.
    exts = {".py", ".js", ".ts", ".cpp", ".c", ".go", ".rs", ".java"}
    files = [
        str(f.relative_to(repo_dir))
        for f in sorted(repo_dir.rglob("*"))[:2000]
        if f.suffix in exts and ".git" not in f.parts
    ]
    if files:
        parts += ["## Source files (sample)", "\n".join(files[:60]), ""]

    return "\n".join(parts)
