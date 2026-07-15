"""RepoRadar MCP server — repo-aware paper search inside MCP clients.

Exposes RepoRadar to Claude Code / Cursor / VS Code / Windsurf over the Model
Context Protocol (stdio). The differentiator vs. the many arXiv MCP servers: these
tools are grounded in *this repository's* profile and ranking, not a generic
search. Run with ``rr mcp``.

The MCP SDK is an optional extra (``pip install 'reporadar[mcp]'``) and is imported
lazily, so this module and the data-gathering helpers below import (and test)
without it — only ``build_server``/``run_stdio`` need it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reporadar.config import ProfilerConfig, RankingConfig
from reporadar.profiler import profile_repo
from reporadar.ranker import format_score_explanation
from reporadar.search import search_corpus
from reporadar.store import PaperStore

# ── Pure helpers (no MCP SDK) — the tool bodies, unit-testable directly ──────


def profile_payload(
    repo_path: str | Path, profiler_cfg: ProfilerConfig | None = None
) -> dict[str, Any]:
    """The repo's inferred topic profile: keywords, libraries, domains."""
    prof = profile_repo(Path(repo_path), profiler_cfg=profiler_cfg)
    return {
        "keywords": [[term, round(weight, 4)] for term, weight in prof.keywords],
        "anchors": list(prof.anchors),
        "domains": list(prof.domains),
    }


def _paper_brief(s: dict[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id": s["arxiv_id"],
        "title": s.get("title"),
        "url": s.get("url"),
        "score_total": s.get("score_total"),
        "llm_score": s.get("llm_score"),
        "llm_reason": s.get("llm_reason"),
        "abstract": (s.get("abstract") or "")[:500],
    }


def ranked_papers_payload(store: PaperStore, limit: int = 10) -> dict[str, Any]:
    """Top-ranked papers from the latest ``rr update`` run (already best-first)."""
    run = store.get_last_run()
    if run is None:
        return {"run_id": None, "papers": [], "note": "No runs yet — run `rr update` first."}
    scores = store.get_scores_for_run(run["run_id"])[: max(0, limit)]
    return {"run_id": run["run_id"], "papers": [_paper_brief(s) for s in scores]}


def explain_relevance_payload(
    store: PaperStore, arxiv_id: str, ranking_cfg: RankingConfig
) -> dict[str, Any]:
    """Why a paper was ranked for this repo — score breakdown + any LLM reason."""
    run = store.get_last_run()
    if run is None:
        return {"error": "No runs yet — run `rr update` first."}
    want = arxiv_id.split("v")[0]
    match = next(
        (s for s in store.get_scores_for_run(run["run_id"]) if s["arxiv_id"].split("v")[0] == want),
        None,
    )
    if match is None:
        return {"error": f"{arxiv_id} is not in the latest run's ranked papers."}
    return {
        "arxiv_id": match["arxiv_id"],
        "title": match.get("title"),
        "explanation": format_score_explanation(match, ranking_cfg),
        "llm_score": match.get("llm_score"),
        "llm_reason": match.get("llm_reason"),
    }


def rate_paper_action(store: PaperStore, arxiv_id: str, rating: int) -> dict[str, Any]:
    """Record a 1–5 usefulness rating (feeds the ranking feedback loop)."""
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        return {"error": "rating must be an integer from 1 (not useful) to 5 (very useful)."}
    store.save_rating(arxiv_id, rating)
    return {"ok": True, "arxiv_id": arxiv_id, "rating": rating}


def search_corpus_payload(store: PaperStore, query: str, limit: int = 10) -> dict[str, Any]:
    """Free-text BM25 search over every paper ever fetched (not just the latest run)."""
    results = search_corpus(store.get_all_papers(), query, limit=max(0, limit))
    return {
        "query": query,
        "count": len(results),
        "papers": [
            {
                "arxiv_id": p["arxiv_id"],
                "title": p.get("title"),
                "url": p.get("url"),
                "published": (p.get("published") or "")[:10],
                "search_score": p.get("search_score"),
                "abstract": (p.get("abstract") or "")[:500],
            }
            for p in results
        ],
    }


# ── MCP server (needs the optional `mcp` SDK) ───────────────────────────────


def build_server(
    repo_path: str | Path,
    db_path: str | Path,
    profiler_cfg: ProfilerConfig | None = None,
    ranking_cfg: RankingConfig | None = None,
) -> Any:
    """Build a FastMCP server exposing RepoRadar's repo-aware tools. Raises
    ImportError if the ``mcp`` extra is not installed."""
    from mcp.server.fastmcp import FastMCP

    ranking = ranking_cfg or RankingConfig()
    server = FastMCP("reporadar")

    @server.tool()
    def get_repo_profile() -> dict[str, Any]:
        """This repository's inferred topic profile: keyword weights, imported
        libraries, and inferred research domains."""
        return profile_payload(repo_path, profiler_cfg)

    @server.tool()
    def get_ranked_papers(limit: int = 10) -> dict[str, Any]:
        """The top arXiv papers RepoRadar ranked for this repository in its most
        recent update, best-first (LLM-triage/hybrid order when enabled)."""
        with PaperStore(db_path) as store:
            return ranked_papers_payload(store, limit)

    @server.tool()
    def explain_relevance(arxiv_id: str) -> dict[str, Any]:
        """Explain why a specific paper (by arXiv id) was ranked for this repo:
        the per-component score breakdown plus any LLM actionability reason."""
        with PaperStore(db_path) as store:
            return explain_relevance_payload(store, arxiv_id, ranking)

    @server.tool()
    def rate_paper(arxiv_id: str, rating: int) -> dict[str, Any]:
        """Record a 1–5 usefulness rating for a paper; ratings tune RepoRadar's
        ranking weights over time."""
        with PaperStore(db_path) as store:
            return rate_paper_action(store, arxiv_id, rating)

    @server.tool()
    def search_papers(query: str, limit: int = 10) -> dict[str, Any]:
        """Free-text search across EVERY paper RepoRadar has fetched for this repo
        (the whole local corpus, not just the latest run), ranked by BM25."""
        with PaperStore(db_path) as store:
            return search_corpus_payload(store, query, limit)

    return server


def run_stdio(
    repo_path: str | Path,
    db_path: str | Path,
    profiler_cfg: ProfilerConfig | None = None,
    ranking_cfg: RankingConfig | None = None,
) -> None:
    """Run the RepoRadar MCP server over stdio (blocks)."""
    build_server(repo_path, db_path, profiler_cfg, ranking_cfg).run()
