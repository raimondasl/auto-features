"""RepoRadar MCP server — repo-aware paper search inside MCP clients.

Exposes RepoRadar to Claude Code / Cursor / VS Code / Windsurf over the Model
Context Protocol (stdio). The differentiator vs. the many arXiv MCP servers: these
tools are grounded in *this repository's* profile and ranking, not a generic
search. Run with ``rr mcp``.

The MCP SDK is an optional extra (``pip install 'reporadar-papers[mcp]'``) and is imported
lazily, so this module and the data-gathering helpers below import (and test)
without it — only ``build_server``/``run_stdio`` need it.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from reporadar.config import (
    DEFAULT_CONFIG_NAME,
    OutputConfig,
    ProfilerConfig,
    RankingConfig,
    TriageConfig,
    default_config_yaml,
    load_config,
    measured_config_yaml,
)
from reporadar.paper_id import dedup_id
from reporadar.profiler import profile_repo
from reporadar.provenance import found_by
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
        # Which retrieval channel contributed it. `dense_discovery` means keyword search did
        # not have it -- the difference between "HyDE found this" and "keyword search would
        # have anyway", which an agent cannot otherwise tell and a user cannot check without
        # querying the store by hand.
        "found_by": found_by(s.get("matched_query")),
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
        "found_by": found_by(match.get("matched_query")),
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
    corpus = store.get_all_papers()
    results = search_corpus(corpus, query, limit=max(0, limit))
    return {
        "query": query,
        "count": len(results),
        # How much there was to search. A caller that gets three hits cannot otherwise tell
        # a narrow corpus from a narrow query, and those call for opposite next moves.
        "corpus_size": len(corpus),
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


# The one line both config templates carry, and the one field no benchmark number
# justifies. `setup_repo` rewrites it in place rather than round-tripping the YAML,
# because the comments above it carry the measurement behind every other value.
_CATEGORIES_LINE = "  categories: [cs.LG, cs.CL]"


@dataclass
class McpReporter:
    """The pipeline's progress, on its way to an MCP client as `notifications/progress`.

    Holds no SDK reference on purpose: *emit* is a plain synchronous callable, so this
    stays importable and unit-testable without the ``mcp`` extra, and the async hop lives
    in ``build_server`` where the Context does.

    The heartbeat is load-bearing rather than decorative. VS Code applies no timeout to a
    tool call at all, but Copilot CLI's 180 s per-request timeout is reset by every
    progress notification and has no absolute cap — so a collection that reports nothing
    for three minutes is cancelled, and one that narrates every stage runs to completion.
    """

    emit: Callable[[int, str], None]
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def info(self, message: str) -> None:
        self._record(message)

    def warn(self, message: str) -> None:
        """Kept separately from `info`, because the caller has to be able to find it.

        The pipeline is deliberately loud when a configured stage cannot run — "HyDE
        discovery unavailable", a source that failed — and never degrades to the
        keyword-only path silently, because that path measured 0/24. But a warning
        forwarded as one more progress line is a status update among sixty, gone the
        moment the next one replaces it. `update_corpus` returns these separately so the
        agent can say what did not run, rather than reporting a thin digest as the state
        of the literature.
        """
        text = self._record(message)
        if text:
            self.warnings.append(text)

    def _record(self, message: str) -> str:
        text = message.strip()
        if not text:
            return ""
        self.messages.append(text)
        # Narration must never be able to destroy what it is narrating. Sending progress
        # fails for reasons that have nothing to do with the work — no request context, a
        # client that went away, a closed pipe — and losing a status line costs a status
        # line, while letting it escape costs minutes of network and LLM calls that had
        # already succeeded.
        with contextlib.suppress(Exception):
            self.emit(len(self.messages), text)
        return text


# The gate block the measured template ships with. `setup_repo` swaps it when the key the
# user actually has is an OpenAI one -- writing `provider: claude` to somebody who was told
# "one OpenAI key is enough" produces a config that demands a credential nobody asked them
# for, and the failure arrives minutes later as "no Claude API key" during collection.
_CLAUDE_GATE = """  provider: claude
  claude_api_key: ${ANTHROPIC_API_KEY}
  claude_model: claude-haiku-4-5"""

# The OpenAI gate as NR-63 measured it -- model and effort included, because "provider:
# openai" alone would silently fall back to the `gpt-4o-mini` default, which is not the
# configuration any published number describes.
_OPENAI_GATE = """  provider: openai
  openai_api_key: ${OPENAI_API_KEY}
  openai_model: gpt-5.6-luna
  openai_reasoning_effort: "none"
  claude_model: claude-haiku-4-5"""


def preferred_provider() -> str:
    """The gate provider to configure, given the credentials this machine actually has.

    OpenAI when neither is present, because the fine-scale rescore is OpenAI-only whatever
    the gate does -- so one OpenAI key runs the whole pipeline, while claude would need two.
    """
    from reporadar import credentials

    if credentials.resolve_api_key("openai"):
        return "openai"
    if credentials.resolve_api_key("claude"):
        return "claude"
    return "openai"


def root_uri_to_path(uri: str) -> Path | None:
    """A `file://` root URI as a local path, or None for anything else.

    Clients send roots as URIs, and on Windows that means `file:///C:/Users/...` -- a
    leading slash before the drive letter that `Path` alone gets wrong. `url2pathname`
    handles it, and percent-escapes (a space in a folder name) come out too.
    """
    parsed = urlparse(str(uri))
    if parsed.scheme != "file":
        return None
    try:
        return Path(url2pathname(unquote(parsed.path)))
    except (OSError, ValueError):  # pragma: no cover - malformed URI
        return None


def choose_repo_root(candidates: list[Path], cwd: Path) -> Path | None:
    """Which of the client's roots this server should treat as the repository.

    A multi-root workspace offers several and the protocol does not say which is "current",
    so: prefer the one that CONTAINS the process working directory -- the deepest such, for
    nested roots -- because that is the one the editor most likely started us for. Failing
    that take the single root if there is only one, and otherwise the first, leaving the
    caller to say the choice was ambiguous rather than pretending it was not.
    """
    usable = [c for c in candidates if c.is_dir()]
    if not usable:
        return None

    containing = []
    for root in usable:
        try:
            cwd.relative_to(root)
        except ValueError:
            continue
        containing.append(root)
    if containing:
        return max(containing, key=lambda r: len(r.parts))
    return usable[0]


# What a checkout looks like from the outside. Deliberately broad and cheap: the question
# is only "could this plausibly be somebody's project", and a false yes costs a confirmation
# while a false no costs an unnecessary question.
_PROJECT_MARKERS = (
    ".git",
    ".hg",
    ".svn",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "CMakeLists.txt",
    "Makefile",
)


def looks_like_a_project(path: Path) -> bool:
    """Whether *path* plausibly holds source somebody works on."""
    return any((path / marker).exists() for marker in _PROJECT_MARKERS)


def looks_like_a_plugin_install(path: Path) -> bool:
    """Whether *path* is an installed agent plugin's own directory.

    Worth naming separately because it is the failure that actually happened: an editor
    launches a plugin's MCP server with the plugin's own folder as the working directory --
    which the Agent Plugins spec mandates as the default -- so a server that trusted its
    CWD profiled the plugin instead of the user's code. Saying "this is a plugin
    installation" is a great deal more use than "this is not a project".
    """
    return (path / "plugin.json").is_file() and (path / ".mcp.json").is_file()


def not_configured_payload(config_path: Path) -> dict[str, Any]:
    """What every tool returns when the repository has no config yet.

    A RESULT rather than a crash. `rr mcp` used to exit 1 here with the explanation on
    stderr, which no MCP client shows anyone — the user saw "server failed to start" and
    the one command that would fix it never reached them.
    """
    return {
        "status": "not_configured",
        "config_path": str(config_path),
        # The directory is reported, not assumed. Until the server asks the client for its
        # roots, this is whatever the editor set as the process CWD -- which is not
        # necessarily the project the user is looking at, and has in practice been the
        # plugin's own install directory. Surfacing it lets the caller catch that.
        "repo_path": str(config_path.parent),
        "why": (
            "This repository has no RepoRadar configuration yet, so there is nothing to "
            "read and no corpus to search."
        ),
        "retry": {"tool": "setup_repo", "with": {}},
    }


def setup_repo_action(
    repo_path: Path,
    config_path: Path,
    *,
    categories: list[str] | None = None,
    measured: bool = True,
    provider: str | None = None,
) -> dict[str, Any]:
    """Create `.reporadar.yml` and `.reporadar/` for this repository.

    Returns `needs_input` rather than guessing when *categories* is absent. `cs.LG, cs.CL`
    is a guess that fits an ML repository and no other, and a wrong list quietly starves
    every stage downstream — so the caller is handed this repository's own inferred
    profile and asked to choose. An agent reading a profile is a better interview than a
    default nobody opens the file to change.
    """
    if config_path.exists():
        return {
            "status": "already_configured",
            "config_path": str(config_path),
            "note": "Left as it is. Edit the file directly to change it.",
        }

    if not categories:
        profile = profile_payload(repo_path, None)
        return {
            "status": "needs_input",
            "missing": ["categories"],
            "why": (
                "arxiv.categories decides what gets collected at all. The cs.LG/cs.CL "
                "default fits an ML repository and no other; on the wrong field it is the "
                "difference between a digest and noise."
            ),
            "repo_profile": profile,
            "retry": {
                "tool": "setup_repo",
                "with": {"categories": ["<arXiv category ids for this repo's field>"]},
            },
        }

    body = measured_config_yaml() if measured else default_config_yaml()
    if _CATEGORIES_LINE not in body:  # pragma: no cover - template drift guard
        raise RuntimeError(
            "the config template no longer contains the categories line this rewrites"
        )
    rendered = "[" + ", ".join(categories) + "]"
    body = body.replace(_CATEGORIES_LINE, f"  categories: {rendered}", 1)

    gate = provider or preferred_provider()
    if measured and gate == "openai" and _CLAUDE_GATE in body:
        body = body.replace(_CLAUDE_GATE, _OPENAI_GATE, 1)

    config_path.write_text(body, encoding="utf-8")
    (repo_path / ".reporadar").mkdir(parents=True, exist_ok=True)
    return {
        "status": "ok",
        "config_path": str(config_path),
        "repo_path": str(repo_path),
        "categories": list(categories),
        "measured": measured,
        "gate_provider": gate,
        # Said out loud so the agent can pass it on. A gate configured for a key the user
        # does not have fails minutes into collection, and the message names a vendor they
        # were never asked for.
        "gate_key_present": bool(_resolved_key(gate)),
        "next": (
            "Call update_corpus to collect and rank papers."
            if _resolved_key(gate)
            else f"No {gate} key found. The user must run `rr auth --provider {gate}` "
            f"themselves — never ask them to paste a key into the chat. Collection will "
            f"still run, but with no actionability gate, which measured net@2 -11."
        ),
    }


def _resolved_key(provider: str) -> str:
    from reporadar import credentials

    return credentials.resolve_api_key(provider)


def collect_payload(
    cfg: Any,
    *,
    repo: Path,
    config_path: Path,
    db: Path,
    report: Any,
) -> dict[str, Any]:
    """Run the collection for this repository, wherever it has to happen.

    `update_corpus`'s body, out here where it can be tested: building the server needs the
    `mcp` extra, which CI does not install, and the branch that matters most is the one that
    only runs when something has gone wrong.

    The server installs as `reporadar-papers[mcp]` and cannot run dense discovery —
    sentence-transformers pulls torch in, and putting that in the plugin would make every
    installation pay gigabytes for a channel most never turn on. So when the configuration
    asks for HyDE and this environment cannot provide it, the identical pipeline runs in a
    `uvx` environment that can (see `reporadar.delegate`).

    **A delegated run that fails falls back here rather than failing the call.** The
    subprocess is the most machinery in the plugin and the likeliest thing to break; a
    keyword-only digest the caller has been told is keyword-only is worth more than an
    error, and the warning is what keeps that honest.
    """
    from reporadar import delegate

    def _here(why: str) -> dict[str, Any]:
        from reporadar.pipeline import run_pipeline

        result = run_pipeline(cfg, repo_path=repo, db_path=db, report=report)
        return {
            "run_id": result.run_id,
            "stopped": result.stopped,
            "queries": len(result.queries),
            "papers": len(result.papers),
            "scored": len(result.scores),
            "collected_in": why,
        }

    plan = delegate.plan(cfg, repo=repo, config_path=config_path, db=db)
    if plan.warning:
        report.warn(f"  {plan.warning}")
    if not plan.delegated:
        return _here(f"this server's environment ({plan.reason})")

    try:
        counts = delegate.run(plan, report=report)
    except delegate.DelegationError as exc:
        report.warn(
            f"  Dense discovery could not run in its own environment, so this collection "
            f"used keyword retrieval only: {exc}"
        )
        return _here("this server's environment, after the delegated run failed")
    counts["collected_in"] = f"{plan.spec}, via uvx ({plan.reason})"
    return counts


def require_sdk() -> None:
    """Import the optional MCP SDK so an unusable one fails here, with its own message.

    ``build_server`` imports the SDK lazily, so a *missing* extra and an *incompatible*
    one both surface as ImportError deep inside serving. Calling this first keeps that
    distinction legible: mcp 2.x renamed FastMCP to MCPServer and its ModuleNotFoundError
    names the migration, which is the text a user actually needs.
    """
    from mcp.server.fastmcp import FastMCP  # noqa: F401


def build_server(
    repo_path: str | Path,
    db_path: str | Path | None = None,
    profiler_cfg: ProfilerConfig | None = None,
    ranking_cfg: RankingConfig | None = None,
    output_cfg: OutputConfig | None = None,
    triage_cfg: TriageConfig | None = None,
    config_path: str | Path | None = None,
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

    server = FastMCP("reporadar")
    _explicit = any(c is not None for c in (ranking_cfg, output_cfg, triage_cfg))
    _cwd_repo = Path(repo_path)
    # Set by `setup_repo(repo_path=...)` and remembered for the rest of the session, so a
    # caller only has to say it once. The last resort that does not depend on the client
    # implementing roots: the agent is editing the project, so it knows where the project
    # is, and telling us directly beats any amount of inference.
    _told: list[Path] = []

    class _Location(NamedTuple):
        repo: Path
        config_path: Path
        db: Path
        source: str

    async def _locate() -> _Location:
        """Which repository this call is about — asked of the CLIENT, not the process.

        An editor launches the plugin's server with a working directory of its own
        choosing, and in practice that has been the plugin's own install directory. So
        inferring the project from the CWD profiled the plugin instead of the user's code,
        and wrote the configuration there too — which is worse than failing, because the
        digest still looks like an answer.

        MCP roots is the protocol's own answer to "which project am I in". The CWD stays as
        the fallback for clients that send none, which is also what keeps `rr mcp` working
        from a terminal.
        """
        # `source` says WHY, not just what. Falling back is a legitimate outcome -- a
        # terminal `rr mcp` has no client roots -- but "cwd" alone cannot distinguish a
        # client that offered nothing from one that was never asked, and that is exactly
        # the question left open when this landed and the directory was still wrong.
        if _told:
            told = _told[-1]
            return _Location(
                repo=told,
                config_path=Path(config_path) if config_path else told / DEFAULT_CONFIG_NAME,
                db=Path(db_path) if db_path else told / ".reporadar" / "papers.db",
                source="told by the caller",
            )

        repo, source = _cwd_repo, "cwd (not determined)"
        try:
            import anyio
            from mcp.types import ClientCapabilities, RootsCapability

            session = server.get_context().session
            # ASK WHETHER IT CAN ANSWER FIRST. `roots/list` is a request to the client, and
            # a client that never replies leaves the tool call hanging for as long as the
            # caller will wait -- every tool, not just this one. Caught by the smoke check,
            # which is a deliberately minimal client and does not implement roots.
            if not session.check_client_capability(ClientCapabilities(roots=RootsCapability())):
                source = "cwd (client declares no roots capability)"
            else:
                # A deadline even when it says it can: a capability is a promise, not a
                # guarantee, and the fallback is always available.
                with anyio.fail_after(10):
                    result = await session.list_roots()
                raw = [str(r.uri) for r in result.roots]
                offered = [p for p in (root_uri_to_path(u) for u in raw) if p]
                chosen = choose_repo_root(offered, _cwd_repo)
                if chosen is not None:
                    repo = chosen
                    source = (
                        "client root"
                        if len(offered) == 1
                        else f"client root (1 of {len(offered)} offered)"
                    )
                elif not raw:
                    source = "cwd (client offered no roots)"
                else:
                    # Roots arrived but none was usable: a non-file scheme, or a path that
                    # does not exist on this machine. Quote them, because at that point the
                    # URIs themselves are the evidence.
                    source = f"cwd (no usable root among {raw[:3]})"
        except TimeoutError:
            source = "cwd (client did not answer roots/list in 10s)"
        except Exception as exc:  # noqa: BLE001 - a client without roots is not an error
            source = f"cwd (roots lookup failed: {type(exc).__name__}: {exc})"[:200]

        return _Location(
            repo=repo,
            config_path=Path(config_path) if config_path else repo / DEFAULT_CONFIG_NAME,
            # An explicit --db wins; otherwise the store belongs to the repository just
            # resolved, not to whatever directory this process was started in.
            db=Path(db_path) if db_path else repo / ".reporadar" / "papers.db",
            source=source,
        )

    def _wrong_place(loc: _Location) -> dict[str, Any] | None:
        """A refusal when the server cannot tell which repository it is meant to serve.

        The old behaviour was to guess from the working directory and carry on, which for a
        plugin install meant profiling the plugin and presenting the result as an answer.
        Guessing is the bug. When the client has not said which project it means AND the
        directory shows no sign of being one, the honest move is to stop and ask -- the
        caller is an agent working inside the project, so it knows the path and only has to
        be asked for it.

        A working directory that DOES look like a project is accepted without ceremony:
        that is `rr mcp` run from a terminal, where the CWD is exactly right.
        """
        if not loc.source.startswith("cwd") or looks_like_a_project(loc.repo):
            return None
        plugin = looks_like_a_plugin_install(loc.repo)
        return {
            "status": "needs_input",
            "missing": ["repo_path"],
            "repo_path": str(loc.repo),
            "repo_source": loc.source,
            "why": (
                (
                    "This is an installed plugin's own directory, not a repository. Editors "
                    "launch a plugin's MCP server here by default."
                    if plugin
                    else "This directory shows no sign of being a project: no .git, no manifest."
                )
                + " The client did not say which repository it means, so there is nothing to "
                "infer from — and a digest built for the wrong repository still looks like an "
                "answer, which is why this stops rather than guessing."
            ),
            "retry": {
                "tool": "setup_repo",
                "with": {"repo_path": "<absolute path to the user's project>"},
            },
        }

    def _full_config(cfg_path: Path) -> Any | None:
        """The whole configuration, or None when the repository has none yet."""
        if not cfg_path.exists():
            return None
        return load_config(cfg_path)

    def _sections(cfg_path: Path) -> tuple[Any, Any, Any] | None:
        """(ranking, output, triage) for THIS call, or None if the repo is unconfigured.

        Read per call rather than once at startup, because `setup_repo` can create the
        config mid-session: a server that decided at boot that the repository was
        uninitialised would keep saying so until somebody restarted it. Explicitly-passed
        sections still win, so callers that hand over config (tests) are unchanged.
        """
        if _explicit:
            return (
                ranking_cfg or RankingConfig(),
                output_cfg or OutputConfig(),
                triage_cfg or TriageConfig(),
            )
        cfg = _full_config(cfg_path)
        if cfg is None:
            return None
        return (cfg.ranking, cfg.output, cfg.triage)

    @server.tool()
    async def get_repo_profile() -> dict[str, Any]:
        """This repository's inferred topic profile: keyword weights, imported
        libraries, and inferred research domains.

        Also reports which directory it profiled and how that was decided — check it
        against the project the user means before trusting anything downstream of it.
        """
        _log_call("get_repo_profile")
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        payload = profile_payload(loc.repo, profiler_cfg)
        payload["repo_path"] = str(loc.repo)
        payload["repo_source"] = loc.source
        return payload

    @server.tool()
    async def get_ranked_papers(limit: int = 10) -> dict[str, Any]:
        """The papers RepoRadar recommends for this repository from its most recent
        update, best-first — the same set and order `rr digest` shows."""
        _log_call("get_ranked_papers", limit=limit)
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        sections = _sections(loc.config_path)
        if sections is None:
            return not_configured_payload(loc.config_path)
        _, output, triage = sections
        with PaperStore(loc.db) as store:
            return ranked_papers_payload(
                store,
                limit,
                repo_path=loc.repo,
                top_n=output.top_n,
                triage_threshold=(triage.min_actionable if triage.enabled else None),
                rerank=(triage.rerank if triage.enabled else False),
                finescale_configured=(
                    triage.finescale.threshold if triage.finescale.enabled else None
                ),
            )

    @server.tool()
    async def explain_relevance(arxiv_id: str) -> dict[str, Any]:
        """Explain why a specific paper (by arXiv id) was ranked for this repo:
        the per-component score breakdown plus any LLM actionability reason."""
        _log_call("explain_relevance", arxiv_id=arxiv_id)
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        sections = _sections(loc.config_path)
        if sections is None:
            return not_configured_payload(loc.config_path)
        ranking, _, _ = sections
        with PaperStore(loc.db) as store:
            return explain_relevance_payload(store, arxiv_id, ranking)

    @server.tool()
    async def rate_paper(arxiv_id: str, rating: int) -> dict[str, Any]:
        """Record a 1–5 usefulness rating for a paper; ratings tune RepoRadar's
        ranking weights over time."""
        _log_call("rate_paper", arxiv_id=arxiv_id, rating=rating)
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        if _sections(loc.config_path) is None:
            return not_configured_payload(loc.config_path)
        with PaperStore(loc.db) as store:
            return rate_paper_action(store, arxiv_id, rating)

    @server.tool()
    async def search_papers(query: str, limit: int = 10) -> dict[str, Any]:
        """Free-text search across EVERY paper RepoRadar has fetched for this repo
        (the whole local corpus, not just the latest run), ranked by BM25."""
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        if _sections(loc.config_path) is None:
            return not_configured_payload(loc.config_path)
        with PaperStore(loc.db) as store:
            payload = search_corpus_payload(store, query, limit)
        # The result count and the corpus size travel with the call. "How wide was this
        # server's corpus" is otherwise only answerable by finding the store on disk and
        # hoping it has not been rebuilt since -- and it is the whole variable in the P27
        # wide-corpus arm.
        _log_call(
            "search_papers",
            query=query,
            limit=limit,
            n_results=payload["count"],
            corpus=payload["corpus_size"],
        )
        return payload

    @server.tool()
    async def setup_repo(
        categories: list[str] | None = None,
        measured: bool = True,
        provider: str | None = None,
        repo_path: str | None = None,
    ) -> dict[str, Any]:
        """Initialise RepoRadar in this repository: write `.reporadar.yml` and `.reporadar/`.

        Call with no arguments first. It answers with this repository's inferred profile and
        asks for `categories`, because arxiv.categories decides what gets collected at all
        and the cs.LG/cs.CL default fits an ML repository and no other. Propose categories
        from the profile, confirm them with the user, then call again with them.

        Check the `repo_path` it reports before confirming: that is the directory which will
        be configured, and it is not always the one the user has in mind.

        If it is wrong — a plugin install directory, an editor folder, anywhere that is not
        the user's project — pass `repo_path` with the correct absolute path. The server
        remembers it for the rest of the session, so every later tool call uses it too.
        """
        _log_call("setup_repo", categories=categories, measured=measured)
        if repo_path:
            told = Path(repo_path).expanduser().resolve()
            if not told.is_dir():
                return {
                    "status": "bad_repo_path",
                    "repo_path": str(told),
                    "why": "that path is not a directory on this machine",
                }
            _told.append(told)
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        result = setup_repo_action(
            loc.repo,
            loc.config_path,
            categories=categories,
            measured=measured,
            provider=provider,
        )
        result["repo_path"] = str(loc.repo)
        result["repo_source"] = loc.source
        return result

    @server.tool()
    async def update_corpus() -> dict[str, Any]:
        """Collect, rank and gate papers for this repository — the pipeline `rr update` runs.

        Minutes rather than seconds, and it reports progress as it goes. Call it once after
        `setup_repo`, and again when you want fresh candidates; `get_ranked_papers` reads
        what this leaves behind and never collects on its own.

        With dense discovery enabled it runs the pipeline in a separate `uvx` environment
        that has the embedding model's dependencies, which this server deliberately does
        not. The first such run is several minutes longer while that environment is built;
        it is cached afterwards. `collected_in` in the result says which happened.
        """
        _log_call("update_corpus")
        # Fetched rather than taken as a parameter: `from __future__ import annotations`
        # turns signatures into strings, and FastMCP evaluates them against MODULE globals
        # -- where a `Context` imported inside this function does not exist. Asking the
        # server for it also keeps `ctx` out of the tool's public schema.
        ctx = server.get_context()
        loc = await _locate()
        if (refusal := _wrong_place(loc)) is not None:
            return refusal
        cfg = _full_config(loc.config_path)
        if cfg is None:
            return not_configured_payload(loc.config_path)

        import anyio
        import anyio.lowlevel

        def emit(n: int, message: str) -> None:
            # Hops from the pipeline's worker thread back onto the event loop. Every one of
            # these also resets Copilot CLI's per-request timeout, so the narration is what
            # keeps a multi-minute collection from being cancelled underneath itself.
            # McpReporter suppresses failures here: `report_progress` raises outright when
            # there is no request context.
            anyio.from_thread.run(ctx.report_progress, float(n), None, message)

        reporter = McpReporter(emit=emit)

        def _collect() -> dict[str, Any]:
            # Everything -- including the decision about WHERE to collect -- happens on this
            # worker thread, because the reporter's progress hop only works from one: a
            # warning emitted on the event loop would be recorded and never sent.
            return collect_payload(
                cfg,
                repo=loc.repo,
                config_path=loc.config_path,
                db=loc.db,
                report=reporter,
            )

        # `run_pipeline` is synchronous and `ctx.report_progress` is not, so the pipeline
        # runs in a worker thread and the reporter hops back. Calling it inline would block
        # the event loop and no progress notification could leave while it ran.
        result = await anyio.to_thread.run_sync(_collect)
        # A CHECKPOINT, and it is load-bearing. When the client cancels this call -- VS Code's
        # Stop button, a TypeScript client's request timeout -- the SDK answers "Request
        # cancelled" at once and marks the request complete, but `run_sync` shields this
        # handler, so the cancellation is deferred until the collection's thread returns and
        # the next await gives it somewhere to land. There was no next await: this returned a
        # dict, the SDK tried to send a SECOND response, and `AssertionError: Request already
        # responded to` killed the session. Silently -- the process stayed up, and the user's
        # next request was simply never answered. Yielding here lets the cancellation raise
        # inside the handler, where the SDK suppresses the duplicate response. It changes
        # nothing for a call that was not cancelled. tests/test_mcp_cancellation.py.
        await anyio.lowlevel.checkpoint()

        return {
            "status": "stopped" if result["stopped"] else "ok",
            "stopped": result["stopped"],
            "run_id": result["run_id"],
            "repo_path": str(loc.repo),
            "queries": result["queries"],
            "papers": result["papers"],
            "scored": result["scored"],
            # WHERE the pipeline ran, always. Dense discovery needs an environment this
            # server does not have, so a collection may have happened in a `uvx` one
            # instead — and "was HyDE actually running?" is the first question asked of a
            # thin digest. It is not answerable afterwards from anything the run leaves
            # behind, so it travels with the result.
            "collected_in": result["collected_in"],
            # Separate from `progress` deliberately: a stage that was configured and could
            # not run is the difference between a thin digest and a thin literature, and
            # the agent has to be able to tell the user which one it is looking at.
            "warnings": reporter.warnings,
            "progress": reporter.messages,
        }

    return server


def run_stdio(
    repo_path: str | Path,
    db_path: str | Path | None = None,
    profiler_cfg: ProfilerConfig | None = None,
    ranking_cfg: RankingConfig | None = None,
    output_cfg: OutputConfig | None = None,
    triage_cfg: TriageConfig | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Run the RepoRadar MCP server over stdio (blocks)."""
    build_server(
        repo_path,
        db_path,
        profiler_cfg,
        ranking_cfg,
        output_cfg,
        triage_cfg,
        config_path=config_path,
    ).run()
