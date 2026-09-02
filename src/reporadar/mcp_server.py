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

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reporadar.config import OutputConfig, ProfilerConfig, RankingConfig, TriageConfig
from reporadar.paper_id import dedup_id
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
    brief = {
        "arxiv_id": s["arxiv_id"],
        "title": s.get("title"),
        "url": s.get("url"),
        "score_total": s.get("score_total"),
        "llm_score": s.get("llm_score"),
        "llm_reason": s.get("llm_reason"),
        "abstract": (s.get("abstract") or "")[:500],
    }
    # An agent acting on a retracted result is the exact harm the withdrawal signal
    # exists to prevent, and an agent never sees the digest's warning section — so the
    # flag has to travel with the paper itself. Absent unless positively flagged.
    if s.get("withdrawn_in"):
        brief["withdrawn"] = True
        brief["warning"] = (
            "The authors withdrew this paper (notice in its "
            f"{s['withdrawn_in']}). Treat its claims as retracted."
        )
    # Why a paper is in `muted` has to travel with it. Without this an agent sees a
    # high-scoring paper set aside for no stated reason and has to guess -- measured
    # 2026-09-01, Opus 5 given this payload excluded the paper and said so, which is the
    # right call made blind. "Already cited" is a *reason*, not a defect in the paper.
    if s.get("already_cited"):
        brief["already_cited"] = True
        brief["note"] = (
            "This repository's own README, CITATION file or bibliography already cites "
            "this paper, so it is not a new recommendation."
        )
    return brief


def ranked_papers_payload(
    store: PaperStore,
    limit: int = 10,
    *,
    repo_path: str | Path | None = None,
    top_n: int = 15,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_configured: float | None = None,
) -> dict[str, Any]:
    """The papers RepoRadar recommends from the latest ``rr update`` run, best-first.

    **Through `categorize_papers`, which is what makes this the same answer the digest
    gives.** It was `get_scores_for_run(run_id)[:limit]` — the raw heuristic/RRF order —
    so an agent asking for "the top papers RepoRadar ranked" got a materially different
    set from the one `rr digest` shows the same user for the same run: no actionability
    gate, no fine-scale bar, no rerank, and withdrawn and already-cited papers still
    occupying slots. The agent got the *weaker* set, and by a wide margin — on the
    benchmark the gate is where the precision comes from, and the heuristic 0.5 threshold
    it replaced measured net@2 −11.

    `archive`, `notify`, `watch`, `rr explain` and the digest itself already share this
    function; this was the one consumer that had its own rule. Same defect shape as C-9,
    C-12 and C-14, and the same fix.

    *limit* is applied AFTER tiering: `top_n` decides what RepoRadar was willing to
    display at all, `limit` how many of those the caller wants. A limit above `top_n`
    therefore cannot reach past the window — a paper outside it is one the product
    declined to show, and a tool call must not be able to promote it.

    The second tier and the muted papers travel too, under their own keys and only when
    non-empty. An agent that never hears about a retraction it might otherwise have found
    on its own is worse off than one told not to use it, and `maybe_relevant` is
    explicitly *not* a recommendation — which is exactly why it must not be merged into
    `papers`.
    """
    run = store.get_last_run()
    if run is None:
        return {"run_id": None, "papers": [], "note": "No runs yet — run `rr update` first."}
    from reporadar.digest import categorize_papers
    from reporadar.finescale import threshold_for_run
    from reporadar.profiler import cited_arxiv_ids_of

    scored = store.get_scores_for_run(run["run_id"])
    top_picks, maybe_relevant, muted = categorize_papers(
        scored,
        top_n=top_n,
        # The same exclusion the digest applies, from a file scan rather than a full
        # profile -- `notify` reads it the same way and for the same reason. On five of
        # six scientific repositories measured, the paper the gate ranked first was the
        # repository's OWN publication.
        cited_ids=cited_arxiv_ids_of(Path(repo_path)) if repo_path is not None else None,
        triage_threshold=triage_threshold,
        rerank=rerank,
        # Derived from the RUN rather than taken from the config, so this matches the
        # digest the user is looking at whether or not the stage ran that run: scores are
        # persisted only when the gate applies, so their presence answers it exactly.
        finescale_threshold=(
            threshold_for_run(scored, finescale_configured)
            if finescale_configured is not None
            else None
        ),
    )
    payload: dict[str, Any] = {
        "run_id": run["run_id"],
        "papers": [_paper_brief(p) for p in top_picks[: max(0, limit)]],
    }
    if maybe_relevant:
        payload["maybe_relevant"] = [_paper_brief(p) for p in maybe_relevant]
    if muted:
        payload["muted"] = [_paper_brief(p) for p in muted]
    return payload


def explain_relevance_payload(
    store: PaperStore, arxiv_id: str, ranking_cfg: RankingConfig
) -> dict[str, Any]:
    """Why a paper was ranked for this repo — score breakdown + any LLM reason."""
    run = store.get_last_run()
    if run is None:
        return {"error": "No runs yet — run `rr update` first."}
    want = dedup_id(arxiv_id)
    match = next(
        (s for s in store.get_scores_for_run(run["run_id"]) if dedup_id(s["arxiv_id"]) == want),
        None,
    )
    if match is None:
        return {"error": f"{arxiv_id} is not in the latest run's ranked papers."}
    payload = {
        "arxiv_id": match["arxiv_id"],
        "title": match.get("title"),
        "explanation": format_score_explanation(match, ranking_cfg),
        "llm_score": match.get("llm_score"),
        "llm_reason": match.get("llm_reason"),
    }
    if match.get("withdrawn_in"):
        payload["withdrawn"] = True
        payload["warning"] = (
            "The authors withdrew this paper (notice in its "
            f"{match['withdrawn_in']}). Its score is penalized and its claims are retracted."
        )
    return payload


def rate_paper_action(store: PaperStore, arxiv_id: str, rating: int) -> dict[str, Any]:
    """Record a 1–5 usefulness rating (feeds the ranking feedback loop)."""
    if not isinstance(rating, int) or not 1 <= rating <= 5:
        return {"error": "rating must be an integer from 1 (not useful) to 5 (very useful)."}
    # Resolve against the stored corpus (version-insensitively, since agents pass
    # unversioned ids) like `rr rate` does. Rating an unknown id would otherwise
    # create an orphan row that other features seed from — e.g. SPECTER2 would
    # then try to cache a vector for a paper that doesn't exist and hit the
    # paper_embeddings foreign key.
    want = dedup_id(arxiv_id)
    stored = store.get_paper(arxiv_id)
    if stored is None:
        stored = next((p for p in store.get_all_papers() if dedup_id(p["arxiv_id"]) == want), None)
    if stored is None:
        return {"error": f"{arxiv_id} is not in this repo's paper store — nothing to rate."}
    resolved = str(stored["arxiv_id"])
    store.save_rating(resolved, rating)
    return {"ok": True, "arxiv_id": resolved, "rating": rating}


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

# Opt-in call log: set RR_MCP_CALL_LOG to a path and every tool call appends one JSON line.
#
# Off unless the variable is set, and it names a file rather than defaulting to one, because
# a server that writes to a user's repository by default is doing something they did not ask
# for. What it exists to answer is the question ROADMAP 2 cannot answer today and cannot
# answer by reasoning: **do agents actually call these tools, and which ones?** A tool an
# agent never discovers is indistinguishable, from the outside, from a tool that did not
# help -- and the two call for opposite responses.
#
# Never raises into a tool call. A telemetry failure that broke the tool it is measuring
# would be worse than no telemetry.
_CALL_LOG_ENV = "RR_MCP_CALL_LOG"


def _log_call(tool: str, **params: Any) -> None:
    path = os.environ.get(_CALL_LOG_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "t": datetime.now(UTC).isoformat(),
                        # The server process's identity. A client that retries spawns a
                        # FRESH server against the same log path, so without this a
                        # retried run's calls are silently pooled with the failed
                        # attempt's -- and a tool-use count is the covariate that decides
                        # whether a null result means "did not help" or "never found".
                        # Wrong data wearing the shape of right data.
                        "pid": os.getpid(),
                        "tool": tool,
                        **params,
                    }
                )
                + "\n"
            )
    except OSError:
        pass


def build_server(
    repo_path: str | Path,
    db_path: str | Path,
    profiler_cfg: ProfilerConfig | None = None,
    ranking_cfg: RankingConfig | None = None,
    output_cfg: OutputConfig | None = None,
    triage_cfg: TriageConfig | None = None,
) -> Any:
    """Build a FastMCP server exposing RepoRadar's repo-aware tools. Raises
    ImportError if the ``mcp`` extra is not installed.

    *output_cfg* and *triage_cfg* are what let `get_ranked_papers` answer with the same
    set `rr digest` shows: the window width, the actionability threshold and whether the
    rerank is on are all configuration, and reading them here rather than defaulting them
    is the difference between "RepoRadar's recommendations" and "a fixed guess at them".
    The gate threshold is applied only when triage is ENABLED — a repo that never ran the
    gate has no `llm_score` on any paper, and filtering on a column that is null
    everywhere would return an empty list rather than the ranking it does have.
    """
    from mcp.server.fastmcp import FastMCP

    ranking = ranking_cfg or RankingConfig()
    output = output_cfg or OutputConfig()
    triage = triage_cfg or TriageConfig()
    server = FastMCP("reporadar")

    @server.tool()
    def get_repo_profile() -> dict[str, Any]:
        """This repository's inferred topic profile: keyword weights, imported
        libraries, and inferred research domains."""
        _log_call("get_repo_profile")
        return profile_payload(repo_path, profiler_cfg)

    @server.tool()
    def get_ranked_papers(limit: int = 10) -> dict[str, Any]:
        """The papers RepoRadar recommends for this repository from its most recent
        update, best-first — the same set and order `rr digest` shows."""
        _log_call("get_ranked_papers", limit=limit)
        with PaperStore(db_path) as store:
            return ranked_papers_payload(
                store,
                limit,
                repo_path=repo_path,
                top_n=output.top_n,
                triage_threshold=(triage.min_actionable if triage.enabled else None),
                rerank=(triage.rerank if triage.enabled else False),
                finescale_configured=(
                    triage.finescale.threshold if triage.finescale.enabled else None
                ),
            )

    @server.tool()
    def explain_relevance(arxiv_id: str) -> dict[str, Any]:
        """Explain why a specific paper (by arXiv id) was ranked for this repo:
        the per-component score breakdown plus any LLM actionability reason."""
        _log_call("explain_relevance", arxiv_id=arxiv_id)
        with PaperStore(db_path) as store:
            return explain_relevance_payload(store, arxiv_id, ranking)

    @server.tool()
    def rate_paper(arxiv_id: str, rating: int) -> dict[str, Any]:
        """Record a 1–5 usefulness rating for a paper; ratings tune RepoRadar's
        ranking weights over time."""
        _log_call("rate_paper", arxiv_id=arxiv_id, rating=rating)
        with PaperStore(db_path) as store:
            return rate_paper_action(store, arxiv_id, rating)

    @server.tool()
    def search_papers(query: str, limit: int = 10) -> dict[str, Any]:
        """Free-text search across EVERY paper RepoRadar has fetched for this repo
        (the whole local corpus, not just the latest run), ranked by BM25."""
        _log_call("search_papers", query=query, limit=limit)
        with PaperStore(db_path) as store:
            return search_corpus_payload(store, query, limit)

    return server


def run_stdio(
    repo_path: str | Path,
    db_path: str | Path,
    profiler_cfg: ProfilerConfig | None = None,
    ranking_cfg: RankingConfig | None = None,
    output_cfg: OutputConfig | None = None,
    triage_cfg: TriageConfig | None = None,
) -> None:
    """Run the RepoRadar MCP server over stdio (blocks)."""
    build_server(repo_path, db_path, profiler_cfg, ranking_cfg, output_cfg, triage_cfg).run()
