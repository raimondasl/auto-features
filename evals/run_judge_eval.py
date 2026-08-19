"""Tier B: LLM-judged, abstention-aware actionable-improvement benchmark.

For each real repo, two systems produce a paper list:
  1. RepoRadar  — its "Top Picks" tier (score >= 0.5), which may be empty.
  2. Baseline   — Opus 4.8 via Claude Code headless (the strong baseline).

A neutral OpenAI judge (default GPT-5.5), blind to the source, scores the pooled
union of both lists 0-3 for whether each paper could genuinely improve THIS repo.
Metrics reward precision and correct abstention and penalize false positives.

    # dry-run the whole pipeline with no keys/spend (mock judge + mock baseline):
    uv run python evals/run_judge_eval.py --mock

    # the real thing (needs OPENAI_API_KEY for the judge):
    #   baseline via the Anthropic API (no Claude Code CLI needed; needs ANTHROPIC_API_KEY):
    uv run python evals/run_judge_eval.py --case rag --baseline api
    #   baseline via Claude Code headless (needs `claude` on PATH):
    uv run python evals/run_judge_eval.py --case rag --baseline cli

See evals/README.md for keys and cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
import judge as judge_mod  # noqa: E402
from harness import (  # noqa: E402
    EVALS_DIR,
    WORK_DIR,
    assemble_repo_context,
    clone_repo,
    collect_live_papers,
    load_benchmark,
)
from metrics import summarize_system  # noqa: E402
from verify import resolve_references  # noqa: E402

from reporadar.collector import CollectionError  # noqa: E402
from reporadar.config import (  # noqa: E402
    ABSENT_CATEGORY_MODES,
    BIGRAM_MODES,
    QueriesConfig,
    RankingConfig,
)
from reporadar.digest import TOP_THRESHOLD  # noqa: E402
from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.ranker import rank_papers  # noqa: E402
from reporadar.retrieval import hybrid_reorder  # noqa: E402
from reporadar.triage import rerank_by_actionability  # noqa: E402

RESULTS_DIR = EVALS_DIR / "results"
ENV_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"]
RECENT_DAYS = 180  # baseline papers newer than this count as "recent"
# --rr-rerank triages this many candidates (vs the top-10) so the llm_score
# reorder can pull a buried-but-actionable paper up into the returned Top-10.
RERANK_POOL = 20
# Cases where the HyDE arm degraded to the keyword pool. A run that quietly lost the
# channel it exists to measure would be reported as a clean arm; this makes it visible.
HYDE_FAILURES: list[str] = []
# --rr-sweep re-gates Top Picks at each of these min_actionable thresholds. Triage
# scores are computed once, so every threshold is free (no extra model calls).
SWEEP_THRESHOLDS = (1, 2, 3)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def case_profile(
    repo_dir: Path,
    *,
    scan_source: bool,
    typed_anchors: bool = False,
    prose_chars: int | None = None,
) -> Any:
    """The one place this harness builds a repository profile.

    Four stages need one — collection, ranking, the gate and the rescore — and they must
    agree about what the profiler is allowed to read. An arm where retrieval sees source
    code and the gate does not is not a configuration anyone ships; it is two experiments
    averaged and reported as one. `digest_window` and `dedup_id` have a single home for
    the same reason, and `tests/test_eval_scan_source.py` forbids a fifth site from
    growing its own.

    *prose_chars* is left at the profiler's default for the retrieval stages and set from
    `--rr-prose-chars` for the two LLM stages, because it governs only how much README
    prose reaches a prompt — the split predates this helper and is deliberate.
    """
    from reporadar.config import ProfilerConfig
    from reporadar.profiler import profile_repo

    cfg = (
        ProfilerConfig(scan_source=scan_source, typed_anchors=typed_anchors)
        if prose_chars is None
        else ProfilerConfig(
            scan_source=scan_source, typed_anchors=typed_anchors, prose_chars=prose_chars
        )
    )
    if typed_anchors:
        # `complete()` reads the credential off whatever object it is handed, and the
        # product loader mirrors it onto ProfilerConfig. Constructing the dataclass
        # directly skips that, so the harness does the same mirroring explicitly rather
        # than letting extraction fail into an empty anchor list and silently produce
        # the control arm.
        cfg.claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        cfg.claude_model = cfg.typed_anchors_model
    return profile_repo(repo_dir, profiler_cfg=cfg)


def collect_candidates(
    repo_dir: Path,
    categories: list[str],
    sources: list[str],
    keys: dict[str, str],
    *,
    all_time: bool = False,
    hyde_cfg: Any | None = None,
    bigrams: str = "adjacent",
    scan_source: bool = False,
    typed_anchors: bool = False,
) -> list[dict[str, Any]]:
    """Everything retrieval found, before any ranking — the unit a frozen pool stores.

    Split out from :func:`reporadar_ranked` so `--rr-frozen-pool` can freeze *candidates*
    rather than a ranked list. Freezing the ranked output made every ranking experiment
    collect live at the 1.04 floor, which is the opposite of what the flag exists for: the
    pool is the dominant variance term and ranking is deterministic given it, so a ranking
    arm is exactly the case freezing should make cheap and sensitive.
    """
    profile = case_profile(repo_dir, scan_source=scan_source)
    lookback = 36500 if all_time else 90
    sort_by = "relevance" if all_time else "submitted"
    papers = collect_live_papers(
        profile,
        categories,
        sources=sources,
        keys=keys,
        lookback_days=lookback,
        sort_by=sort_by,
        bigrams=bigrams,
    )
    if hyde_cfg is not None:
        papers = _add_hyde_candidates(papers, profile, keys, hyde_cfg)
    return papers


def rank_candidates(
    repo_dir: Path,
    papers: list[dict[str, Any]],
    categories: list[str],
    top_n: int = 10,
    *,
    all_time: bool = False,
    hybrid: bool = False,
    absent_category: str = "omit",
    scan_source: bool = False,
    typed_anchors: bool = False,
    w_embedding: float = 0.0,
) -> list[tuple[dict[str, Any], float]]:
    """RepoRadar's ranking over an already-collected pool: top-N (paper, score) best-first.

    Deterministic given *papers*, which is what makes a frozen candidate pool sound: the
    same pool re-ranked under the same flags reproduces the stored run exactly, and under
    different flags isolates the ranking change with the collection noise removed.

    The profile is recomputed rather than stored — it is local, free, and a function of
    flags the pool fingerprint already covers.
    """
    if not papers:
        return []
    profile = case_profile(repo_dir, scan_source=scan_source)
    lookback = 36500 if all_time else 90
    w_recency = 0.0 if all_time else 0.3
    ranking_cfg = RankingConfig(
        w_keyword=1.0,
        w_category=0.5,
        w_recency=w_recency,
        absent_category=absent_category,
        w_embedding=w_embedding,
    )
    repo_embedding = None
    if w_embedding > 0:
        # VOID, NOT NULL. Without the extra the ranker scores every paper on keyword and
        # category alone, so the arm would report "the embedding weight does nothing"
        # about a component that never ran -- the exact shape of C-9 and NR-30.
        from reporadar.embeddings import EMBEDDINGS_AVAILABLE, compute_repo_embedding

        if not EMBEDDINGS_AVAILABLE:
            raise SystemExit(
                "--rr-w-embedding needs the `embeddings` extra; refusing to run an arm "
                "whose treatment would silently be absent"
            )
        repo_embedding = compute_repo_embedding(repo_dir)
        if repo_embedding is None:
            raise SystemExit(f"no repo text to embed for {repo_dir.name}: weight is inert")
    ranked = rank_papers(
        papers,
        profile,
        ranking_cfg,
        QueriesConfig(),
        categories or ["cs.LG"],
        lookback_days=lookback,
        repo_embedding=repo_embedding,
    )
    if hybrid:
        ranked = hybrid_reorder(ranked, papers, profile)
    by_id = {p["arxiv_id"]: p for p in papers}
    out: list[tuple[dict[str, Any], float]] = []
    for s in ranked[:top_n]:
        if s["arxiv_id"] in by_id:
            out.append((by_id[s["arxiv_id"]], s["score_total"]))
    return out


def reporadar_ranked(
    repo_dir: Path,
    categories: list[str],
    sources: list[str],
    keys: dict[str, str],
    top_n: int = 10,
    *,
    all_time: bool = False,
    hybrid: bool = False,
    hyde_cfg: Any | None = None,
    bigrams: str = "adjacent",
    absent_category: str = "omit",
    scan_source: bool = False,
    typed_anchors: bool = False,
    w_embedding: float = 0.0,
) -> list[tuple[dict[str, Any], float]]:
    """RepoRadar's real ranking: top-N (paper, score) best-first.

    Top Picks (the abstention-respecting output) = those with score >= 0.5.
    Returning the top-N regardless lets us tell a conservative threshold apart
    from genuinely shallow ranking.

    ``all_time=True`` switches discovery from the 90-day recency window to an
    all-time, relevance-sorted fetch and drops the recency weight from ranking,
    so seminal older papers can surface and compete on relevance alone. This
    tests whether the baseline's edge is a discovery-window artifact.

    ``hybrid=True`` fuses the heuristic ranking with a BM25 lexical ranking via
    RRF before the top-N cut, so a paper the keyword ranker buried on vocabulary
    mismatch can still surface (roadmap #4).

    Kept as the collect-then-rank composition so existing callers and diagnostics read
    unchanged; the two halves are separately callable for frozen-pool runs.
    """
    papers = collect_candidates(
        repo_dir,
        categories,
        sources,
        keys,
        all_time=all_time,
        hyde_cfg=hyde_cfg,
        bigrams=bigrams,
        scan_source=scan_source,
    )
    return rank_candidates(
        repo_dir,
        papers,
        categories,
        top_n=top_n,
        all_time=all_time,
        hybrid=hybrid,
        absent_category=absent_category,
        scan_source=scan_source,
        w_embedding=w_embedding,
    )


def _load_goals(path: str | None) -> dict[str, str] | None:
    """Case -> maintainer goal statement, for the stated-intent arm (roadmap item 0)."""
    if not path:
        return None
    goals: dict[str, str] = json.loads(Path(path).read_text(encoding="utf-8"))
    return goals


def _hyde_cfg(args: argparse.Namespace, goal: str | None = None) -> dict[str, Any] | None:
    """The HyDE arm's settings, or None when the flag is off."""
    if not args.rr_hyde:
        return None
    return {
        "index_dir": Path(args.rr_hyde_index).expanduser(),
        "model": args.rr_triage_model,
        "n_hypotheses": args.rr_hyde_hypotheses,
        "top_k": args.rr_hyde_top_k,
        "verify": not args.rr_hyde_skip_verify,
        "goal": goal,
    }


def _add_hyde_candidates(
    papers: list[dict[str, Any]], profile: Any, keys: dict[str, str], hyde_cfg: Any
) -> list[dict[str, Any]]:
    """Extend the candidate pool with the shipped HyDE channel, before ranking.

    Routed through `reporadar.hyde` and `reporadar.collector.collect_by_ids` rather than
    reimplemented — the same discipline `--rr-finescale` follows, and for the same reason:
    a harness that rebuilds the thing under test measures the harness. Here it matters
    doubly, because the hypothesis prompt is what the 27/48 was measured with.

    A HyDE failure is reported and the run continues on the keyword pool alone. That is a
    DEGRADED arm, not a clean one — the whole point of the flag is the extra candidates —
    so it prints loudly and the summary counts how many cases it happened to.
    """
    from reporadar import hyde
    from reporadar.collector import collect_by_ids

    cfg = SimpleNamespace(
        provider="claude",
        claude_api_key=keys.get("ANTHROPIC_API_KEY", ""),
        claude_model=hyde_cfg["model"],
        timeout=120,
    )
    try:
        ids = hyde.discover(
            profile,
            cfg,
            hyde_cfg["index_dir"],
            n_hypotheses=hyde_cfg["n_hypotheses"],
            top_k=hyde_cfg["top_k"],
            verify=hyde_cfg["verify"],
            goal=hyde_cfg.get("goal"),
        )
    except Exception as exc:  # noqa: BLE001 — a degraded arm must be visible, not fatal
        print(f"        !! HyDE FAILED, continuing on the keyword pool alone: {exc}")
        HYDE_FAILURES.append(str(exc))
        return papers

    # `dedup_id`, not a local `split("v")[0]`. Both normalise an arXiv id and they disagree
    # on the old-style ones (`cs/0602007v4`), five of which sit in this benchmark's judged
    # pools — so which rule a call site picked was itself a silent divergence. See
    # `evals/audit_product_divergence.py`.
    known = {dedup_id(p["arxiv_id"]) for p in papers}
    fresh = [pid for pid in ids if pid not in known]
    try:
        extra = collect_by_ids(fresh)
    except CollectionError as exc:
        print(f"        !! HyDE metadata fetch failed: {exc}")
        HYDE_FAILURES.append(str(exc))
        return papers
    added = [p for p in extra if dedup_id(p["arxiv_id"]) not in known]
    print(
        f"        HyDE: {len(ids)} candidates, {len(fresh)} new, {len(added)} resolved "
        f"(pool {len(papers)} -> {len(papers) + len(added)})"
    )
    return papers + added


def sweep_top_picks(
    ranked: list[dict[str, Any]],
    gains_of: Callable[[list[dict[str, Any]]], list[int]],
    pool_gains: list[int],
    thresholds: tuple[int, ...] = SWEEP_THRESHOLDS,
) -> dict[int, dict[str, Any]]:
    """Top Picks metrics at each ``min_actionable`` threshold, from one triaged run.

    Triage scores (0-3) are computed once, so re-gating the returned set at each
    threshold is free — this exposes the precision/recall trade of raising the
    Top-Pick bar without any extra model calls.
    """
    out: dict[int, dict[str, Any]] = {}
    for t in thresholds:
        picks = [p for p in ranked if (p.get("llm_score") or 0) >= t]
        out[t] = summarize_system(gains_of(picks), pool_gains, n_hallucinated=0)
    return out


def aggregate_sweep(
    per_case: list[dict[int, dict[str, Any]]],
    thresholds: tuple[int, ...] = SWEEP_THRESHOLDS,
) -> dict[int, dict[str, Any]]:
    """Cross-case rollup of a threshold sweep. Per threshold: mean net@2, how many
    cases abstain (returned=0), and how many emit a *false positive* (returned>0
    but 0 actionable — e.g. the webdev negative control leaking a Top Pick)."""
    summary: dict[int, dict[str, Any]] = {}
    for t in thresholds:
        ms = [c[t] for c in per_case if t in c]
        n = len(ms)
        nets = [m["net_value@2"] for m in ms]
        precs = [m["precision"] for m in ms if m["n_returned"] > 0]
        summary[t] = {
            "n_cases": n,
            "mean_net@2": sum(nets) / n if n else 0.0,
            "n_abstained": sum(1 for m in ms if m["n_returned"] == 0),
            "n_false_positive": sum(
                1 for m in ms if m["n_returned"] > 0 and m["n_actionable"] == 0
            ),
            "mean_precision": sum(precs) / len(precs) if precs else float("nan"),
        }
    return summary


def _triage_reporadar(
    repo_dir: Path,
    papers: list[dict[str, Any]],
    keys: dict[str, str],
    model: str,
    prose_chars: int = 300,
    scan_source: bool = False,
    typed_anchors: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run Feature 6 LLM triage over RepoRadar's ranked papers (Claude/Anthropic).

    *prose_chars* is the README budget on the profile; 0 withholds it. The prompt itself
    is always the shipped one — this used to assemble its own "README context" variant,
    and that copy is exactly how a measurement got published under the wrong name: it
    read ``_collect_text_corpus(repo)[0]``, which is the packaging one-liner on 11 of the
    12 benchmark repos, and it silently dropped the domains/key-topics block as well. A
    harness that rebuilds the prompt measures the harness. See evals/RESULTS.md.
    """
    from reporadar.config import SuggestionsConfig
    from reporadar.triage import triage_papers

    profile = case_profile(repo_dir, scan_source=scan_source, prose_chars=prose_chars)
    llm_cfg = SuggestionsConfig(
        provider="claude", claude_api_key=keys.get("ANTHROPIC_API_KEY", ""), claude_model=model
    )
    return triage_papers(papers, profile, llm_cfg, top_k=len(papers))


def _apply_finescale(
    repo_dir: Path, rr_topn: list[dict[str, Any]], keys: dict[str, str], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Rescore the gate's threshold band and return the surviving Top Picks.

    Routed through ``reporadar.finescale`` and ``reporadar.triage.repo_context_block``
    rather than reimplemented, for the reason this file learned the hard way: a harness
    that rebuilds a prompt measures the harness. It matters more here than anywhere else,
    because the score→probability map is *calibrated to that exact prompt* — a local copy
    that drifted by a newline would silently move where P crosses 2/3.

    Mutates ``rr_topn`` in place with ``finescale``/``finescale_p`` so the per-paper
    values land in the results file and the run can be re-analysed without re-calling.
    """
    from reporadar.finescale import enough_scored, score_papers

    above = [p for p in rr_topn if (p.get("llm_score") or 0) > args.rr_min_actionable]
    band = [p for p in rr_topn if p.get("llm_score") == args.rr_min_actionable]
    if not band:
        print("        finescale: no papers in the gate's threshold band — nothing to rescore")
        return above

    profile = case_profile(
        repo_dir,
        scan_source=args.rr_scan_source,
        typed_anchors=args.rr_typed_anchors,
        prose_chars=args.rr_prose_chars,
    )
    cfg = SimpleNamespace(
        openai_api_key=keys.get("OPENAI_API_KEY", ""),
        openai_model=args.rr_finescale_model,
        timeout=60,
    )
    scored = score_papers(band, profile, cfg)
    for p in band:
        got = scored.get(p["arxiv_id"])
        if got:
            p["finescale"] = got["finescale"]
            p["finescale_p"] = got["finescale_p"]

    if not enough_scored(len(scored), len(band)):
        # Same rule the product applies: a stage that mostly failed must not be allowed
        # to demote the whole band and pass it off as an abstention.
        print(
            f"        !! finescale scored only {len(scored)}/{len(band)} — gate NOT applied "
            f"for this case (check OPENAI_API_KEY)"
        )
        return above + band

    kept = [p for p in band if (p.get("finescale_p") or -1.0) >= args.rr_finescale_threshold]
    print(
        f"        finescale: {len(kept)}/{len(band)} band papers clear "
        f"P >= {args.rr_finescale_threshold:.2f} (+{len(above)} above the band)"
    )
    return above + kept


# Every flag that can change WHICH papers are collected or how they are ranked. A frozen
# pool is only reusable when all of these match; anything downstream of here (the gate
# model, min_actionable, the fine-scale threshold) may vary freely, and varying it is the
# entire point of freezing.
POOL_FLAGS = (
    "sources",
    "rr_all_time",  # sets the lookback window and the arXiv sort order
    "rr_prose_chars",
    "rr_ablate_docs",
    "rr_hyde",
    "rr_hyde_index",
    "rr_hyde_hypotheses",
    "rr_hyde_top_k",
    "rr_triage_model",  # HyDE writes its hypotheses with this model
    "rr_bigrams",  # changes the query strings, therefore the pool
    # Source scanning changes the PROFILE, which changes the queries, which changes the
    # pool. A frozen pool reused across it would compare two arms over one arm's
    # candidates and call the difference a treatment effect.
    "rr_scan_source",
    # Typed anchors change the PROFILE for the same reason source scanning does: they
    # reach keywords, therefore the queries, therefore the pool. P9 measured the size of
    # that change -- keywords move on 17 of 25 cases, arXiv queries on 14.
    "rr_typed_anchors",
)

# Flags that change how the pool is ORDERED, not what is in it. Deliberately absent from
# the fingerprint: varying one against a frozen pool is the point of freezing, and ranking
# is deterministic given the pool, so the same flags reproduce the seeding run exactly.
#
# `rr_all_time` is in POOL_FLAGS above rather than here even though it also sets
# `w_recency`, because it changes the fetch window — a pool collected over 90 days cannot
# answer an all-time question, whatever the ranker then does with it.
RANKING_FLAGS = (
    "rr_pool",
    "rr_rerank",
    "rr_hybrid",
    "rr_absent_category",
    "rr_window",
    # Changes the SCORE, not the candidates: the profile is untouched, so queries and
    # collection are identical and one frozen pool serves both arms.
    "rr_w_embedding",
)

# Frozen pools stored the RANKED top-N until 2026-08-13, which made every ranking
# experiment collect live at the 1.04 floor — the opposite of the flag's purpose. Version 2
# stores the candidates instead. Bumped rather than silently reinterpreted: a v1 file read
# as v2 would hand a 20-paper ranked list to the ranker as if it were the whole pool.
FROZEN_POOL_VERSION = 2


def pool_fingerprint(args: argparse.Namespace, case: dict[str, Any], goal: str | None) -> str:
    """Identity of everything upstream of selection, for --rr-frozen-pool.

    A frozen pool reused under different retrieval settings would be the silent-staleness
    failure this project has already paid for twice (a baseline cache that outlived its
    flags; a verdict cache keyed without the prompt). So the key covers the case, its
    categories, every collection/ranking flag, and the **goal** — a goal changes the HyDE
    hypotheses, so a goal experiment is a retrieval experiment and must collect live.
    """
    parts = [case["name"], ",".join(case.get("expected_categories") or [])]
    for flag in POOL_FLAGS:
        parts.append(f"{flag}={getattr(args, flag, None)!r}")
    parts.append(f"goal={goal!r}")
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def frozen_pool_path(pool_dir: Path, case_name: str) -> Path:
    return pool_dir / f"{case_name}.json"


def load_frozen_pool(
    pool_dir: Path, case_name: str, fingerprint: str
) -> list[dict[str, Any]] | None:
    """The stored candidate pool, or None when absent. Raises on mismatch or old format."""
    path = frozen_pool_path(pool_dir, case_name)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version", 1)
    if version != FROZEN_POOL_VERSION:
        raise SystemExit(
            f"frozen pool {path} is format v{version}; this run needs "
            f"v{FROZEN_POOL_VERSION}.\n"
            "v1 stored the RANKED top-N, v2 stores the candidate pool. Reading a v1 file "
            "as v2 would hand a 20-paper ranked list to the ranker as if it were the whole "
            "pool, and every metric would be computed over a pool that had already been "
            "cut by the settings under test. Re-seed into a fresh directory."
        )
    if data.get("fingerprint") != fingerprint:
        # Two hashes and no explanation is a correct refusal that costs the reader an
        # investigation. Adding `rr_scan_source` to POOL_FLAGS on 2026-08-16 invalidated
        # every frozen pool in the project, and working that out from the hashes alone took
        # three commands. So name the flags the SET gained or lost, when that is the cause.
        stored_flags = data.get("pool_flags")
        detail = ""
        if stored_flags is not None:
            added = [f for f in POOL_FLAGS if f not in stored_flags]
            removed = [f for f in stored_flags if f not in POOL_FLAGS]
            if added or removed:
                detail = (
                    "\nThe POOL_FLAGS *set* changed since this pool was collected"
                    + (f"; added {added}" if added else "")
                    + (f"; removed {removed}" if removed else "")
                    + ".\nA new flag makes every stored pool stale even at its default: the "
                    "pool carries no value for a dimension that did not exist when it was "
                    "collected. That is deliberate. Omitting default-valued flags from the "
                    "hash instead would let a pool collected under an OLD default match a "
                    "run under a NEW one, and this project changes defaults."
                )
        else:
            detail = (
                "\nThis pool predates flag-set recording, so the cause cannot be narrowed "
                "further; a POOL_FLAGS change is the most likely one."
            )
        raise SystemExit(
            f"frozen pool {path} was collected under different retrieval settings\n"
            f"  stored:   {data.get('fingerprint')}  ({data.get('collected_at')})\n"
            f"  this run: {fingerprint}{detail}\n"
            "Reusing it would measure the old settings under the new run's name. Use a "
            "different --rr-frozen-pool directory, or drop the flag to collect live.\n"
            f"Note that ranking flags ({', '.join(RANKING_FLAGS)}) are NOT part of the "
            "fingerprint — varying one against a frozen pool is what freezing is for."
        )
    papers: list[dict[str, Any]] = data["candidates"]
    return papers


def save_frozen_pool(
    pool_dir: Path,
    case_name: str,
    fingerprint: str,
    candidates: list[dict[str, Any]],
) -> None:
    """Store the pool as collected, before ranking.

    Never stores an empty pool: an empty candidate list and a failed collection are the
    same bytes on disk, and a frozen empty would score as a legitimate 0.0 on every later
    run that reused it — the mistake that once cost two benchmark cases.
    """
    if not candidates:
        return
    pool_dir.mkdir(parents=True, exist_ok=True)
    frozen_pool_path(pool_dir, case_name).write_text(
        json.dumps(
            {
                "version": FROZEN_POOL_VERSION,
                "case": case_name,
                "fingerprint": fingerprint,
                # The flag SET the fingerprint was computed over, so a later mismatch can
                # name what changed instead of printing two opaque hashes at the reader.
                "pool_flags": list(POOL_FLAGS),
                "collected_at": datetime.now(UTC).isoformat(),
                "n": len(candidates),
                "candidates": candidates,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")
# Must track `profiler._extract_anchors`. `setup.cfg` joined it when MACE was found
# profiling with zero anchors; a manifest the profiler reads and this list omits makes
# the ablation arm differ from its control in a second way, and the thin-docs result
# would then be measuring the omission.
MANIFESTS = ("requirements.txt", "pyproject.toml", "setup.cfg", "setup.py", "package.json")


def ablate_docs(repo_dir: Path, budget: int, *, scan_source: bool = False) -> Path:
    """A copy of *repo_dir* whose self-description is capped at *budget* characters.

    The benchmark's thinnest README is 1,639 characters against a 300-character prose
    budget — **no case is under 1,000, and none under 300** — so every measurement of
    what to tell the system about a repository was made where supply exceeds demand by
    5.5x. RepoRadar's actual target user is a private codebase with almost no prose.
    This builds that case out of a real one.

    Only the README and ``docs/`` are removed. Dependency manifests are copied verbatim,
    because a repository with no documentation still declares its dependencies, and the
    profile derived from them is the floor this experiment is trying to find.

    **This is faithful only while the profiler reads a bounded document set.** With
    ``scan_source=False`` it reads manifests, packaging metadata, the README and
    ``docs/`` — all copied or deliberately withheld here. Turn source scanning on and
    the copy would silently lose signal a real thin-docs repository *has*, making the
    ablation look worse than the thing it models, so this refuses rather than drifts.

    The copy is minimal by construction (a handful of files) rather than a tree copy,
    and it is written to a scratch directory: the real clones gate the verdict cache and
    are never touched.
    """
    # Reads the ARM's setting, not `ProfilerConfig().scan_source`. The original guard
    # asked the dataclass default, which is False and always was — so the moment
    # `--rr-scan-source` existed, the guard would have gone on passing while the exact
    # incoherence it was written to stop became reachable for the first time. A guard
    # that consults a constant is a guard against nothing.
    if scan_source:
        raise SystemExit(
            "ablate_docs models a thin-docs repo by withholding prose only; with "
            "scan_source on it would also withhold code, which a thin-docs repo has. "
            "Copy the source tree here before running this arm again."
        )

    out = WORK_DIR / "ablated" / f"{repo_dir.name}-b{budget}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for manifest in MANIFESTS:
        src = repo_dir / manifest
        if src.is_file():
            shutil.copy2(src, out / manifest)

    kept = 0
    if budget > 0:
        for name in README_NAMES:
            src = repo_dir / name
            if src.is_file():
                text = src.read_text(encoding="utf-8", errors="ignore")[:budget]
                (out / name).write_text(text, encoding="utf-8")
                kept = len(text)
                break
    print(f"        ablated docs: README {kept} chars, docs/ withheld (budget {budget})")
    return out


STAGE_FIELDS = ("llm_score", "finescale", "finescale_p")


def returned_records(
    papers: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Which papers a system actually returned, with the judge's verdict on each.

    Counts alone make a regression undebuggable. When `speech` fell from net@2 +8.0 to
    −2.0 between two runs, the artifacts could say *how many* papers the triage gate
    admitted and how many the judge rejected, but not *which*, so there was no way to
    tell a bad gate from a different candidate pool without paying to re-run.

    The gate's own scores travel with the verdict for the same reason one step further
    in: *why* a paper was or was not shown is a property of ``llm_score`` and
    ``finescale_p``, and a run that records only the outcome cannot be asked whether the
    probability map was still calibrated without re-scoring every paper. The 2026-08-09
    calibration audit paid exactly that (`evals/calibrate_finescale.py`); it should not
    have had to.

    A stage field is emitted **only when that stage ran on that paper** — absent means
    "not scored", present-and-null means "scoring failed". Collapsing those two into a
    null would make a run in which triage never executed indistinguishable from one in
    which it executed and failed, which is the same distinction
    :func:`reporadar.finescale.score_papers` exists to preserve.
    """
    out = []
    for p in papers:
        v = verdicts.get(dedup_id(p["arxiv_id"])) or {}
        rec: dict[str, Any] = {
            "arxiv_id": p["arxiv_id"],
            "title": p.get("title", ""),
            "judge_score": v.get("score"),
            "judge_justification": v.get("justification", ""),
        }
        rec.update({f: p[f] for f in STAGE_FIELDS if f in p})
        out.append(rec)
    return out


def is_recent(published: str) -> bool:
    if not published:
        return False
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (datetime.now(UTC) - dt).days <= RECENT_DAYS


def run(case: dict, keys: dict[str, str], args: argparse.Namespace) -> dict[str, Any] | None:
    name = case["name"]
    print(f"\n[{name}] {case['live_repo']}")
    dest = clone_repo(case["live_repo"], WORK_DIR / name)
    if dest is None:
        return None

    # The judge's view is built from the REAL repository and stays that way under
    # --rr-ablate-docs. Ablating it too would degrade the ground truth alongside the
    # treatment, and the arm would then measure a confused judge agreeing with a
    # confused system. "Useful for this repository" is a property of the repository.
    repo_context = assemble_repo_context(dest)
    rr_dest = (
        dest
        if args.rr_ablate_docs is None
        else ablate_docs(dest, args.rr_ablate_docs, scan_source=args.rr_scan_source)
    )
    goal = None
    if args.rr_goals is not None:
        if name not in args.rr_goals:
            # Silently running without it would measure the control arm under this arm's
            # name — the degraded-arm failure this harness already guards for HyDE.
            raise SystemExit(f"--rr-goals file has no goal for case {name!r}")
        goal = args.rr_goals[name]
        print(f'        goal: "{goal[:96]}"')
    categories = case["expected_categories"]

    # 1. RepoRadar ranking -> Top-10 (diagnostic) and Top Picks (headline).
    #    --rr-rerank triages a deeper candidate pool (RERANK_POOL) and reorders it
    #    by llm_score before the Top-10 cut, so an actionable paper the heuristic
    #    ranker buried below rank 10 can still rise into the returned set.
    # `--rr-pool` overrides the depth. Rank-stratified labelling showed ranks 1-10 and
    # 11-50 hold statistically identical actionable rates (31% vs 33%), so RERANK_POOL=20
    # stops well inside the flat region and the top-10 cut discards ~13 actionable papers
    # per case. The returned set is still cut at 10, so this tests SELECTION quality at a
    # fixed digest size rather than simply returning more papers.
    candidate_n = args.rr_pool or (RERANK_POOL if args.rr_rerank else 10)
    if args.rr_window > candidate_n:
        # The digest cannot be wider than the ranked list it is cut from, so a window
        # larger than the candidate depth silently produces a NARROWER digest than the
        # flag says — and the run records `digest_window: 15` either way, which is the
        # worst version: the artifact asserts a width the run did not have. It bites
        # exactly when the gate is off, because `candidate_n` then defaults to 10 while
        # `--rr-window` defaults to 15. Refuse rather than warn.
        raise SystemExit(
            f"--rr-window {args.rr_window} exceeds the candidate depth {candidate_n}: the "
            f"digest would be cut at {candidate_n} and recorded as {args.rr_window}.\n"
            f"Pass --rr-pool {args.rr_window} (or more) to rank at least that many."
        )
    # --rr-frozen-pool: reuse one collection across arms so a downstream treatment is not
    # measured through a fresh draw of candidates. Two runs of this identical config
    # overlap only 0.50 by Jaccard on the ranked top-10 (evals/noise_floor.py), which is
    # the single largest variance term in every paired comparison here.
    fingerprint = pool_fingerprint(args, case, goal)
    pool_mode, collected_at = "live", None
    candidates = None
    if args.rr_frozen_pool is not None:
        stored = load_frozen_pool(args.rr_frozen_pool, name, fingerprint)
        if stored is not None:
            candidates = stored
            pool_mode = "frozen"
            collected_at = json.loads(
                frozen_pool_path(args.rr_frozen_pool, name).read_text(encoding="utf-8")
            )["collected_at"]
            print(
                f"        FROZEN POOL: {len(candidates)} candidates reused, NOT collected "
                f"live (collected {collected_at[:19]}, fingerprint {fingerprint})"
            )
    if candidates is None:
        candidates = collect_candidates(
            rr_dest,
            categories,
            args.sources,
            keys,
            all_time=args.rr_all_time,
            hyde_cfg=_hyde_cfg(args, goal),
            bigrams=args.rr_bigrams,
            scan_source=args.rr_scan_source,
            typed_anchors=args.rr_typed_anchors,
        )
        if args.rr_frozen_pool is not None:
            save_frozen_pool(args.rr_frozen_pool, name, fingerprint, candidates)
            pool_mode, collected_at = "frozen-seeded", datetime.now(UTC).isoformat()
            print(
                f"        FROZEN POOL SEEDED: {len(candidates)} candidates written "
                f"({fingerprint}) — this run collected LIVE"
            )
    # Ranking always runs, frozen or not. It is deterministic given the pool, so a reused
    # pool reproduces its seeding run exactly under the same flags — and isolates the
    # change under test when a ranking flag differs, which is the whole point of freezing.
    rr_ranked = rank_candidates(
        rr_dest,
        candidates,
        categories,
        top_n=candidate_n,
        all_time=args.rr_all_time,
        hybrid=args.rr_hybrid,
        absent_category=args.rr_absent_category,
        scan_source=args.rr_scan_source,
        typed_anchors=args.rr_typed_anchors,
        w_embedding=args.rr_w_embedding,
    )
    rr_candidates = [p for p, _ in rr_ranked]
    if args.rr_triage:
        # Feature 6: gate Top Picks on the LLM actionability score instead of the
        # heuristic 0.5 threshold, so the benchmark measures triage's effect.
        triaged = _triage_reporadar(
            rr_dest,
            rr_candidates,
            keys,
            args.rr_triage_model,
            args.rr_prose_chars,
            scan_source=args.rr_scan_source,
            typed_anchors=args.rr_typed_anchors,
        )
        for p in rr_candidates:
            p["llm_score"] = triaged.get(p["arxiv_id"], {}).get("llm_score")
        ordered = rerank_by_actionability(rr_candidates) if args.rr_rerank else rr_candidates
        rr_topn = ordered[: args.rr_window]
        rr_toppicks = [p for p in rr_topn if (p.get("llm_score") or 0) >= args.rr_min_actionable]
        n_scored = sum(1 for p in rr_topn if p.get("llm_score") is not None)
        print(
            f"        RepoRadar[triaged{'+rerank' if args.rr_rerank else ''}]: "
            f"{n_scored}/{len(rr_topn)} scored, {len(rr_toppicks)} actionable "
            f"(Top Picks, min>={args.rr_min_actionable})"
        )
        if args.rr_finescale:
            # The shipped second stage, through the shipped module. Everything measured
            # so far about it was an offline replay of a stored run; this is the live
            # path. Papers sitting exactly AT the gate threshold must also clear the
            # calibrated probability — papers above it are trusted on the gate's word.
            rr_toppicks = _apply_finescale(rr_dest, rr_topn, keys, args)
    else:
        rr_topn = rr_candidates[: args.rr_window]
        rr_toppicks = [p for p, s in rr_ranked[: args.rr_window] if s >= TOP_THRESHOLD]
        print(
            f"        RepoRadar: {len(rr_topn)} ranked, "
            f"{len(rr_toppicks)} in Top Picks tier (>=0.5)"
        )

    # 2. Baseline (Opus, via Claude Code CLI or the Anthropic API) -> verify
    if args.baseline == "none":
        b: dict[str, Any] = {"ids": [], "titles": [], "status": "skipped", "cost_usd": 0.0}
    else:
        b = baseline_mod.run_baseline(
            dest,
            repo_name=name,
            repo_context=repo_context,
            mode=args.baseline,
            mock=args.mock,
            use_cache=not args.no_cache,
        )
    baseline_status = b.get("status", "ok")
    if baseline_status == "skipped":
        print("        baseline skipped (--baseline none) — pool is RepoRadar's top-10 only")
    elif baseline_status != "ok":
        print(f"        !! BASELINE DID NOT RUN [{baseline_status}]: {b.get('raw', '')[:200]}")
    b_papers, n_halluc, n_lookup_failed = resolve_references(b["ids"], b["titles"])
    if n_lookup_failed:
        # An arXiv outage can't be blamed on the baseline — mark it unverified
        # rather than counting real papers as hallucinated.
        baseline_status = "arxiv_unverified" if baseline_status == "ok" else baseline_status
        print(f"        !! arXiv lookup failed for {n_lookup_failed} baseline ref(s) — unverified")
    baseline_ok = baseline_status == "ok"
    print(
        f"        Baseline recommended {len(b['ids']) + len(b['titles'])} ref(s) -> "
        f"{len(b_papers)} real, {n_halluc} hallucinated (cost ${b.get('cost_usd', 0):.2f})"
    )

    # 3. Pool = RepoRadar top-N ∪ baseline; judge each once, blind to source.
    #    A judge failure drops the paper from the pool — never fabricate a 0.
    pool: dict[str, dict[str, Any]] = {}
    for p in rr_topn + b_papers:
        pool.setdefault(dedup_id(p["arxiv_id"]), p)
    verdicts: dict[str, dict[str, Any]] = {}
    n_judge_failed = 0
    for base_id, paper in list(pool.items()):
        try:
            verdicts[base_id] = judge_mod.judge_paper(
                name,
                repo_context,
                paper,
                model=args.model,
                mock=args.mock,
                use_cache=not args.no_cache,
            )
        except Exception as exc:  # noqa: BLE001 — never score an unjudged paper as 0
            n_judge_failed += 1
            pool.pop(base_id, None)
            print(f"        ! judge failed for {base_id} (dropped): {str(exc)[:120]}")
    if n_judge_failed:
        print(f"        ! {n_judge_failed} paper(s) dropped from the pool (judge errors)")

    def gains_for(papers: list[dict[str, Any]]) -> list[int]:
        return [
            int(verdicts[bid]["score"])
            for p in papers
            if (bid := dedup_id(p["arxiv_id"])) in verdicts
        ]

    pool_gains = [int(v["score"]) for v in verdicts.values()]
    rr_pick_metrics = summarize_system(gains_for(rr_toppicks), pool_gains, n_hallucinated=0)
    rr_topn_metrics = summarize_system(gains_for(rr_topn), pool_gains, n_hallucinated=0)

    if baseline_ok:
        # Restrict to papers still in the pool (a dropped one has no gain).
        b_present = [p for p in b_papers if dedup_id(p["arxiv_id"]) in verdicts]
        b_gains = gains_for(b_papers)
        b_metrics = summarize_system(b_gains, pool_gains, n_hallucinated=n_halluc)
        recent_gains = [
            g for p, g in zip(b_present, b_gains, strict=True) if is_recent(p.get("published", ""))
        ]
        b_metrics["n_recent"] = len(recent_gains)
        b_metrics["net_value_recent@2"] = summarize_system(recent_gains, pool_gains)["net_value@2"]
    else:
        # Failed/unverified: emit NO real metric numbers, so no aggregation can
        # read the crash as a legitimate 0.0 net-value / 1.0 abstention.
        b_metrics = {
            "failed": True,
            "n_returned": len(b_papers),
            "n_hallucinated": n_halluc,
            "n_lookup_failed": n_lookup_failed,
        }
    b_metrics["status"] = baseline_status

    n_relevant = sum(1 for g in pool_gains if g >= 2)
    print(f"        pool judged: {len(pool_gains)} papers, {n_relevant} genuinely actionable (>=2)")
    _print_system("RepoRadar[TopPicks]", rr_pick_metrics)
    _print_system("RepoRadar[Top10]   ", rr_topn_metrics)
    b_extra = "" if not baseline_ok else f"recent={b_metrics['n_recent']}/{len(b_papers)}"
    _print_system("Baseline           ", b_metrics, extra=b_extra)

    sweep: dict[int, dict[str, Any]] | None = None
    if args.rr_sweep:
        # Free: re-gate the same triaged Top-10 at every threshold, no new calls.
        sweep = sweep_top_picks(rr_topn, gains_for, pool_gains)
        print("        Top Picks sweep (min_actionable):")
        for t in SWEEP_THRESHOLDS:
            _print_system(f"  min>={t}          ", sweep[t])

    def _returned(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return returned_records(papers, verdicts)

    result: dict[str, Any] = {
        "case": name,
        "repo": case["live_repo"],
        # Recorded on every case so no reader, and no aggregation script, can mistake a
        # frozen-pool arm for a live one. `noise_floor.py` and `ablation_report.py` refuse
        # to compare across modes.
        "pool_provenance": {
            "mode": pool_mode,
            "fingerprint": fingerprint,
            "collected_at": collected_at,
            "pool_dir": str(args.rr_frozen_pool) if args.rr_frozen_pool else None,
        },
        # The phrase-query arm, recorded per case so a three-arm comparison never has to
        # infer the arm from a filename someone may have renamed.
        "bigram_mode": args.rr_bigrams,
        "absent_category": args.rr_absent_category,
        # The remaining two RANKING_FLAGS, for the same reason as every field above it:
        # both change the returned set, and neither was recorded until an ablation was set
        # up whose two arms would have produced run files identical in every recorded
        # field except the numbers — the artifact could not say which arm it was. Added
        # 2026-08-16, before that experiment ran rather than after.
        "hybrid": bool(args.rr_hybrid),
        "rerank": bool(args.rr_rerank),
        # What the profiler was allowed to read. Recorded because it changes the pool,
        # the ranking, the gate prompt and the rescore prompt at once.
        "scan_source": bool(args.rr_scan_source),
        "typed_anchors": bool(args.rr_typed_anchors),
        "w_embedding": float(args.rr_w_embedding),
        # The documentation budget this arm ran under, or None for an unablated run.
        # The last POOL_FLAG that was not recorded: the four arms of the thin-docs
        # grid could only be told apart on 2026-08-16 by matching their means against
        # a derived summary file, so deleting that file would have made four runs
        # mutually unidentifiable. Adding the field changes no past number -- it is an
        # output on new artifacts, not an input to anything -- and the pool fingerprint
        # is untouched because rr_ablate_docs was always in POOL_FLAGS.
        "ablate_docs": args.rr_ablate_docs,
        # How many ranked candidates went forward — the number the gate sees when
        # `--rr-triage` is on, which is what the product spells `triage.top_k`. `pool_size`
        # below is the *judged* pool and `pool_provenance` is where the candidates came
        # from; neither answers "how deep did we gate", and that is both the arm of the
        # depth experiment and a shipped default no measurement has ever included.
        "gate_depth": candidate_n,
        # How many ranked papers could reach a tier — the product spells it
        # `output.top_n`. Recorded so an arm cannot be reported under a window its own
        # run file contradicts.
        "digest_window": args.rr_window,
        "sources": list(args.sources),
        "pool_size": len(pool_gains),
        "n_actionable_in_pool": n_relevant,
        "n_judge_failed": n_judge_failed,
        "baseline_status": baseline_status,
        "reporadar_toppicks": rr_pick_metrics,
        "reporadar_top10": rr_topn_metrics,
        "baseline": b_metrics,
        "returned": {
            "reporadar_toppicks": _returned(rr_toppicks),
            "reporadar_top10": _returned(rr_topn),
            "baseline": _returned(b_papers) if baseline_ok else [],
        },
    }
    if sweep is not None:
        result["reporadar_toppicks_sweep"] = sweep
    return result


def _print_system(label: str, m: dict[str, Any], extra: str = "") -> None:
    if m.get("failed"):
        print(
            f"          {label}: ** FAILED ({m.get('status')}) — no metrics **  "
            f"(returned={m.get('n_returned', 0)}, halluc={m.get('n_hallucinated', 0)})"
        )
        return
    prec = m["precision"]
    prec_s = "n/a(abstained)" if prec != prec else f"{prec:.2f}"  # NaN check
    print(
        f"          {label}: returned={m['n_returned']}  actionable={m['n_actionable']}  "
        f"precision={prec_s}  net@2={m['net_value@2']:+.1f}  ndcg={m['ndcg@k']:.2f}  "
        f"halluc={m['n_hallucinated']}  {extra}".rstrip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", help="Only run this case, or a comma-separated subset of cases.")
    parser.add_argument(
        "--mock", action="store_true", help="Mock judge + baseline (no keys/spend)."
    )
    parser.add_argument(
        "--model", default=judge_mod.DEFAULT_JUDGE_MODEL, help="OpenAI judge model."
    )
    parser.add_argument("--sources", default="arxiv", help="RepoRadar sources (comma-separated).")
    parser.add_argument(
        "--baseline",
        choices=["cli", "api", "none"],
        default="cli",
        help="Opus baseline mode: 'cli' = Claude Code headless (needs `claude` on PATH); "
        "'api' = Anthropic Messages API + web_search (needs ANTHROPIC_API_KEY, no CLI); "
        "'none' = skip it. Use 'none' only for RepoRadar-vs-RepoRadar arm comparisons, "
        "where the baseline contributes nothing but cost — and note it also shrinks the "
        "judged pool to RepoRadar's own top-10, so pool statistics are NOT comparable "
        "with runs that included it.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached verdicts/baseline.")
    parser.add_argument(
        "--rr-triage",
        action="store_true",
        help="Gate RepoRadar Top Picks on Feature 6 LLM triage (needs ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--rr-triage-model", default="claude-haiku-4-5", help="Model for RepoRadar triage."
    )
    parser.add_argument(
        "--rr-min-actionable",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Top-Pick gate: a paper is returned only if its triage llm_score is >= this "
        "(default 2). Raising it trades recall for precision. Implies --rr-triage.",
    )
    parser.add_argument(
        "--rr-sweep",
        action="store_true",
        help="Report Top Picks metrics at every min_actionable threshold (1/2/3) in one run — "
        "free, since triage scores are computed once. Directly shows which gate maximizes net@2 "
        "and eliminates false positives. Implies --rr-triage.",
    )
    parser.add_argument(
        "--rr-rerank",
        action="store_true",
        help=f"Listwise-rerank RepoRadar's Top Picks by LLM actionability: triage a deeper "
        f"pool of {RERANK_POOL} candidates and reorder by llm_score before the Top-10 cut, so a "
        f"buried-but-actionable paper can surface. Implies --rr-triage. Incurs more triage spend.",
    )
    parser.add_argument(
        "--rr-prose-chars",
        type=int,
        default=300,
        help="README budget on the profile the gate sees (profiler.prose_chars). 300 is "
        "the shipped default and the measured optimum (+22 net@2 over 0 on 602 labelled "
        "papers; 2000 and 6000 both score lower). 0 withholds it, which is the "
        "pre-2026-08-02 behaviour and the control arm for any prose measurement.",
    )
    parser.add_argument(
        "--rr-hyde",
        action="store_true",
        help="Add HyDE dense-index candidates to the pool before ranking, via the shipped "
        "reporadar.hyde. Measured in isolation at 27/48 targets with 15 reachable by no "
        "other channel; this flag is how that converts (or does not) into net@2. Needs a "
        "synced index — see --rr-hyde-index.",
    )
    parser.add_argument(
        "--rr-hyde-index",
        default=str(EVALS_DIR / ".work" / "hyde_index"),
        help="Index directory. Defaults to the one P4's replication already built.",
    )
    parser.add_argument(
        "--rr-bigrams",
        choices=BIGRAM_MODES,
        default="verified",
        help="Phrase-query policy, defaulting to the shipped one. `adjacent` pairs each "
        "keyword with its TF-IDF neighbour whether or not the two words belong together — "
        'it built "use page" for duckdb and "data cd" for redis, and it is what every '
        "number published before 2026-08-12 was measured with. `verified` (shipped) keeps "
        "only pairs occurring literally in the repo text. `none` drops phrase queries and "
        "was measured WORSE (-0.48 net@2/case). On 25 arXiv cases all three tie inside the "
        "1.04 floor; the difference shows on keyword sources, which have no category clause "
        "to fall back on. A retrieval setting: it is in POOL_FLAGS.",
    )
    parser.add_argument(
        "--rr-absent-category",
        choices=ABSENT_CATEGORY_MODES,
        default="omit",
        help="How w_category treats a paper with NO categories — every paper from every "
        "non-arXiv source. `omit` (default, shipped) drops the component, which was meant "
        "to avoid handicapping those papers and instead advantages them: at equal keyword "
        "relevance an uncategorised paper scores 0.600 against an arXiv paper's 0.567, and "
        "0.600 against 0.400 when the arXiv paper matches no target category. `zero` "
        "scores the absence as 0; `impute` scores it at the pool's mean category score. A "
        "RANKING flag, so a frozen pool can be reused across values of it.",
    )
    parser.add_argument(
        "--rr-scan-source",
        action="store_true",
        help="Let the profiler read source files as well as prose (the shipped "
        "`profiler.scan_source`, which no benchmark arm has ever enabled). A thin-docs "
        "repository is thin in prose but has code; NR-26 found its richer arm's benefit "
        "tracked that extra information rather than the question asked of it.",
    )
    parser.add_argument(
        "--rr-typed-anchors",
        action="store_true",
        help="Let the profiler read the README with an LLM and merge the named entities "
        "it finds into anchors (the shipped `profiler.typed_anchors`, default off). P9 "
        "measured the channel discriminating where the manifest channel does not and P10 "
        "found that survives redacting the spans from the judge's view; neither showed it "
        "improves a digest, which is what this arm is for. A POOL flag: it changes the "
        "profile, therefore the queries, therefore the pool.",
    )
    parser.add_argument(
        "--rr-w-embedding",
        type=float,
        default=0.0,
        help="Ranking weight for repo/paper embedding similarity. The dataclass default "
        "and every published number are 0.0; the config `rr init` writes is 1.5, larger "
        "than w_keyword, never measured in this role.",
    )
    parser.add_argument("--rr-hyde-hypotheses", type=int, default=4)
    parser.add_argument("--rr-hyde-top-k", type=int, default=100)
    parser.add_argument(
        "--rr-hyde-skip-verify",
        action="store_true",
        help="Skip the bit-exact encoder check. Only for a re-run in the same session that "
        "already verified — a mismatched encoder makes every HyDE number noise.",
    )
    parser.add_argument(
        "--rr-finescale",
        action="store_true",
        help="Apply the shipped fine-scale rescore (reporadar.finescale) to the papers "
        "sitting exactly at --rr-min-actionable: score each 0-9, read the expectation over "
        "the answer token's logprob distribution, and keep only those clearing "
        "--rr-finescale-threshold. Needs OPENAI_API_KEY (Anthropic exposes no logprobs). "
        "Implies --rr-triage. ~$0.01 per case. Measured offline at +1.91 -> +3.14 mean "
        "net@2; this flag is how that gets confirmed on a live run.",
    )
    parser.add_argument(
        "--rr-frozen-pool",
        metavar="DIR",
        type=Path,
        default=None,
        help="Collect each case's ranked candidates once into DIR, then REUSE them. Two runs "
        "of the identical config overlap only 0.50 by Jaccard on the ranked top-10, and that "
        "is the largest variance term in every paired comparison here. MEASURED 2026-08-11: "
        "freezing takes the minimum resolvable effect from 1.04 to 0.48 net@2/case — it HALVES "
        "the floor rather than quartering it. An earlier version of this help claimed ~0.2, "
        "which was an unmeasured guess and wrong by about 2x (see evals/noise_floor.py). "
        "VALID ONLY FOR TREATMENTS DOWNSTREAM OF RETRIEVAL (gate model, min_actionable, the "
        "fine-scale threshold). Anything that changes what gets collected — --rr-hyde, "
        "--rr-all-time, --rr-goals, --rr-ablate-docs — is part of the pool fingerprint and a "
        "mismatch is a hard error, never a silent reuse. Frozen runs are labelled per case in "
        "`pool_provenance`, banner-marked on stdout, and written to a `-frozenpool-` filename; "
        "noise_floor.py and ablation_report.py refuse to compare across modes.",
    )
    parser.add_argument(
        "--rr-goals",
        metavar="FILE",
        help="Stated-intent arm (roadmap item 0): a JSON {case: goal} file, e.g. "
        "evals/goals/blind.json. The goal reaches ONLY the HyDE hypothesis prompt — "
        "reporadar.hyde appends it after the shared repo block, so it structurally cannot "
        "enter the 0-3 gate or the fine-scale rescore. That placement is P8's result, not "
        "a preference: stated wants fed to the GATE scored net@2 +57 against +95, the worst "
        "arm in the campaign, and that experiment concluded wants belong in the query. A "
        "case missing from the file is a hard error, never a silent fallback to control.",
    )
    parser.add_argument(
        "--rr-ablate-docs",
        type=int,
        default=None,
        metavar="CHARS",
        help="Thin-docs arm: build RepoRadar's profile from a repo whose README is capped "
        "at CHARS and whose docs/ is withheld (0 = manifests only). The judge still sees "
        "the REAL repo, so ground truth does not degrade with the treatment. Every case in "
        "the benchmark has a README of 1,639+ chars against a 300-char prose budget, so "
        "nothing here has ever measured the regime RepoRadar's target user lives in. Note "
        "that at CHARS >= 300 the gate's prose block is IDENTICAL to the control's (both "
        "are README[:300]) and only the derived keywords, queries and HyDE hypotheses "
        "thin out — which is what isolates retrieval degradation from prompt degradation.",
    )
    parser.add_argument(
        "--rr-finescale-model",
        default="gpt-4o-mini",
        help="Model for the fine-scale rescore. Must expose logprobs, and must be the one "
        "the shipped probability map was fitted against unless you refit it.",
    )
    parser.add_argument(
        "--rr-finescale-threshold",
        type=float,
        default=2.0 / 3.0,
        help="P(actionable) a band paper must clear. The default is DERIVED, not tuned: "
        "net@2 values a shown paper at 3p-2, so showing pays exactly above 2/3. Moving it "
        "trades recall for precision and makes the run incomparable to the shipped policy.",
    )
    parser.add_argument(
        "--rr-pool",
        type=int,
        default=0,
        help="how many candidates to triage before the Top-10 cut (default: 20 with "
        "--rr-rerank, else 10). Deeper costs more triage calls, not more judge calls, "
        "unless the deeper papers actually reach the returned set.",
    )
    parser.add_argument(
        "--rr-window",
        type=int,
        default=15,
        help="how many ranked papers may reach a tier — the product's `output.top_n`. "
        "Defaulted to 10 until 2026-08-15, when widening it to the shipped 15 measured "
        "+1.24 net@2/case (CI [+0.48, +2.08]) against a 0.48 frozen floor; the benchmark "
        "now describes what ships. RUNS BEFORE THAT DATE ARE WINDOW-10 and not comparable "
        "— every run records `digest_window` so a report can refuse to mix them. Unlike "
        "--rr-pool this DOES cost judge calls: the judged pool is the returned set, so "
        "each extra slot is a new verdict per case.",
    )
    parser.add_argument(
        "--rr-all-time",
        action="store_true",
        help="RepoRadar discovery: all-time relevance-sorted fetch (no 90-day window, "
        "recency weight dropped) so seminal older papers can surface. Tests whether the "
        "baseline's edge is a discovery-window artifact. NOTE: surfaces new papers not in "
        "the judge cache, so this incurs fresh OpenAI judge (and triage) spend.",
    )
    parser.add_argument(
        "--rr-hybrid",
        action="store_true",
        help="Hybrid retrieval (roadmap #4): fuse the heuristic ranking with a BM25 lexical "
        "ranking via RRF before the Top-N cut, so a paper buried on vocabulary mismatch can "
        "surface. Changes the candidate order (may shift which papers are judged/triaged).",
    )
    args = parser.parse_args()
    args.rr_goals = _load_goals(args.rr_goals)
    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.rr_rerank or args.rr_sweep or args.rr_finescale:
        # All three need llm_scores: rerank orders by them, the sweep re-gates on them,
        # and the fine-scale stage only rescores the band the gate itself defines.
        args.rr_triage = True

    load_dotenv(EVALS_DIR / ".env")
    keys = {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}

    if args.rr_finescale and not args.mock and "OPENAI_API_KEY" not in os.environ:
        # Fail here, not 22 cases in: without the key every band paper would fail to
        # score, every case would fall back to un-gated, and the run would look like a
        # legitimate measurement of the old path under a new name.
        raise SystemExit("--rr-finescale needs OPENAI_API_KEY (see evals/README.md)")

    judge_label = "mock" if args.mock else args.model
    baseline_label = "mock" if args.mock else f"claude-opus-4-8 ({args.baseline})"
    rr_gate = (
        f"triage{'+rerank' if args.rr_rerank else ''}({args.rr_triage_model}, "
        f"min>={args.rr_min_actionable}{'+sweep' if args.rr_sweep else ''})"
    )
    rr_label = rr_gate if args.rr_triage else "heuristic 0.5"
    disco_label = "all-time/relevance" if args.rr_all_time else "90-day/recency"
    disco_label += "+hybrid(bm25+rrf)" if args.rr_hybrid else ""
    print("=== RepoRadar Tier B: actionable-improvement benchmark ===")
    print(f"judge={judge_label}  baseline={baseline_label}  reporadar_gate={rr_label}")
    print(f"reporadar_discovery={disco_label}")
    if args.rr_frozen_pool is not None:
        print(
            "*** FROZEN POOL MODE — candidates are reused, NOT collected live. Valid only "
            "for treatments downstream of retrieval; results are NOT comparable with "
            f"live-collection runs. dir={args.rr_frozen_pool} ***"
        )
    print(f"keys present: {', '.join(keys) or 'none'}")
    if not args.mock and "OPENAI_API_KEY" not in keys:
        print("\n! OPENAI_API_KEY not set. Set it (see evals/README.md) or use --mock.")
        return 1
    if not args.mock and args.baseline == "api" and "ANTHROPIC_API_KEY" not in keys:
        print("\n! --baseline api needs ANTHROPIC_API_KEY. Set it (see evals/README.md).")
        return 1
    if args.rr_triage and "ANTHROPIC_API_KEY" not in keys:
        print("\n! --rr-triage needs ANTHROPIC_API_KEY (RepoRadar triage uses Claude).")
        return 1

    bench = load_benchmark()
    WORK_DIR.mkdir(exist_ok=True)
    results = []
    failed_collection: list[str] = []
    only = {c.strip() for c in args.case.split(",") if c.strip()} if args.case else None
    if only:
        # A typo'd name would otherwise silently shrink the run, and a 5-case arm
        # reported as a 6-case arm is how a subset gets compared against a whole.
        unknown = only - {c["name"] for c in bench["cases"]}
        if unknown:
            raise SystemExit(f"--case names not in the benchmark: {sorted(unknown)}")
    for case in bench["cases"]:
        if only and case["name"] not in only:
            continue
        try:
            r = run(case, keys, args)
        except CollectionError as exc:
            # A case whose candidate fetch failed has NO measurement, and the one thing it
            # must not do is contribute a 0.0 to the mean as though the system had honestly
            # returned nothing. `db` and `storage` did exactly that on 2026-08-07 and moved
            # a 22-case mean by nearly a full point.
            print(f"[{case['name']}] EXCLUDED — collection failed, not scored: {exc}")
            failed_collection.append(case["name"])
            continue
        if r:
            results.append(r)

    if failed_collection:
        print(
            f"\n!! {len(failed_collection)} case(s) EXCLUDED for failed collection: "
            f"{', '.join(failed_collection)}"
        )
        print(
            "   Every mean below is over the cases that actually collected. Re-run them "
            "before comparing against another arm."
        )

    if HYDE_FAILURES:
        # Never let a degraded arm be reported as a clean one: if HyDE fell back to the
        # keyword pool on some cases, those cases measure the OLD channel under the new
        # flag's name, which is exactly how a null result gets manufactured.
        print(
            f"\n!! HyDE degraded to the keyword pool on {len(HYDE_FAILURES)} case(s). "
            f"This arm is NOT a clean HyDE measurement."
        )
        for msg in dict.fromkeys(HYDE_FAILURES):
            print(f"   {msg[:160]}")

    if args.rr_sweep and results:
        key = "reporadar_toppicks_sweep"
        per_case = [r[key] for r in results if key in r]
        agg = aggregate_sweep(per_case)
        print(f"\n=== Top Picks threshold sweep — cross-case ({len(per_case)} cases) ===")
        print("  (higher min_actionable = stricter gate: trades recall for precision)")
        for t in SWEEP_THRESHOLDS:
            s = agg[t]
            prec = s["mean_precision"]
            prec_s = "n/a" if prec != prec else f"{prec:.2f}"  # NaN check
            print(
                f"  min>={t}:  mean net@2={s['mean_net@2']:+.2f}   "
                f"abstained={s['n_abstained']}/{s['n_cases']}   "
                f"false-positive={s['n_false_positive']}/{s['n_cases']}   "
                f"mean precision={prec_s}"
            )

    if results and not args.mock:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        # The filename carries the mode so a frozen-pool run cannot be mistaken for a
        # live one at a glance, or picked up by a later script that assumes live.
        tag = "-frozenpool" if args.rr_frozen_pool is not None else ""
        # Same reasoning for the phrase-query arm: three runs of one experiment differ only
        # in this flag, and telling them apart from the filename beats opening each one.
        if args.rr_bigrams != "adjacent":
            tag += f"-bigrams_{args.rr_bigrams}"
        if args.rr_absent_category != "omit":
            tag += f"-abscat_{args.rr_absent_category}"
        # Fusion is on in every headline, so the *absence* is the notable arm — the same
        # convention as `abscat`, which tags the two modes that are not the default.
        if args.rr_triage and not args.rr_hybrid:
            tag += "-nohybrid"
        if args.rr_scan_source:
            tag += "-scansource"
        if args.rr_w_embedding:
            tag += f"-wemb{args.rr_w_embedding:g}"
        out = RESULTS_DIR / f"judge-{args.model}{tag}-{stamp}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
