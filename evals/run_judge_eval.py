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
import json
import os
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
    profile_case_repo,
)
from metrics import summarize_system  # noqa: E402
from verify import resolve_references  # noqa: E402

from reporadar.collector import CollectionError  # noqa: E402
from reporadar.config import QueriesConfig, RankingConfig  # noqa: E402
from reporadar.digest import TOP_THRESHOLD  # noqa: E402
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
    """
    profile = profile_case_repo(repo_dir)
    lookback = 36500 if all_time else 90  # ~100 years = effectively no date cutoff
    sort_by = "relevance" if all_time else "submitted"
    w_recency = 0.0 if all_time else 0.3
    papers = collect_live_papers(
        profile, categories, sources=sources, keys=keys, lookback_days=lookback, sort_by=sort_by
    )
    if hyde_cfg is not None:
        papers = _add_hyde_candidates(papers, profile, keys, hyde_cfg)
    if not papers:
        return []
    ranking_cfg = RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=w_recency)
    ranked = rank_papers(
        papers,
        profile,
        ranking_cfg,
        QueriesConfig(),
        categories or ["cs.LG"],
        lookback_days=lookback,
    )
    if hybrid:
        ranked = hybrid_reorder(ranked, papers, profile)
    by_id = {p["arxiv_id"]: p for p in papers}
    out: list[tuple[dict[str, Any], float]] = []
    for s in ranked[:top_n]:
        if s["arxiv_id"] in by_id:
            out.append((by_id[s["arxiv_id"]], s["score_total"]))
    return out


def _hyde_cfg(args: argparse.Namespace) -> dict[str, Any] | None:
    """The HyDE arm's settings, or None when the flag is off."""
    if not args.rr_hyde:
        return None
    return {
        "index_dir": Path(args.rr_hyde_index).expanduser(),
        "model": args.rr_triage_model,
        "n_hypotheses": args.rr_hyde_hypotheses,
        "top_k": args.rr_hyde_top_k,
        "verify": not args.rr_hyde_skip_verify,
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
        )
    except Exception as exc:  # noqa: BLE001 — a degraded arm must be visible, not fatal
        print(f"        !! HyDE FAILED, continuing on the keyword pool alone: {exc}")
        HYDE_FAILURES.append(str(exc))
        return papers

    known = {p["arxiv_id"].split("v")[0] for p in papers}
    fresh = [pid for pid in ids if pid not in known]
    try:
        extra = collect_by_ids(fresh)
    except CollectionError as exc:
        print(f"        !! HyDE metadata fetch failed: {exc}")
        HYDE_FAILURES.append(str(exc))
        return papers
    added = [p for p in extra if p["arxiv_id"].split("v")[0] not in known]
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
) -> dict[str, dict[str, Any]]:
    """Run Feature 6 LLM triage over RepoRadar's ranked papers (Claude/Anthropic).

    *prose_chars* is the README budget on the profile; 0 withholds it. The prompt itself
    is always the shipped one — this used to assemble its own "README context" variant,
    and that copy is exactly how a measurement got published under the wrong name: it
    read ``_collect_text_corpus(repo)[0]``, which is the packaging one-liner on 11 of the
    12 benchmark repos, and it silently dropped the domains/key-topics block as well. A
    harness that rebuilds the prompt measures the harness. See evals/RESULTS.md.
    """
    from reporadar.config import ProfilerConfig, SuggestionsConfig
    from reporadar.profiler import profile_repo
    from reporadar.triage import triage_papers

    profile = profile_repo(repo_dir, profiler_cfg=ProfilerConfig(prose_chars=prose_chars))
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
    from reporadar.config import ProfilerConfig
    from reporadar.finescale import enough_scored, score_papers
    from reporadar.profiler import profile_repo

    above = [p for p in rr_topn if (p.get("llm_score") or 0) > args.rr_min_actionable]
    band = [p for p in rr_topn if p.get("llm_score") == args.rr_min_actionable]
    if not band:
        print("        finescale: no papers in the gate's threshold band — nothing to rescore")
        return above

    profile = profile_repo(repo_dir, profiler_cfg=ProfilerConfig(prose_chars=args.rr_prose_chars))
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
        v = verdicts.get(p["arxiv_id"].split("v")[0]) or {}
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

    repo_context = assemble_repo_context(dest)
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
    rr_ranked = reporadar_ranked(
        dest,
        categories,
        args.sources,
        keys,
        top_n=candidate_n,
        all_time=args.rr_all_time,
        hybrid=args.rr_hybrid,
        hyde_cfg=_hyde_cfg(args),
    )
    rr_candidates = [p for p, _ in rr_ranked]
    if args.rr_triage:
        # Feature 6: gate Top Picks on the LLM actionability score instead of the
        # heuristic 0.5 threshold, so the benchmark measures triage's effect.
        triaged = _triage_reporadar(
            dest, rr_candidates, keys, args.rr_triage_model, args.rr_prose_chars
        )
        for p in rr_candidates:
            p["llm_score"] = triaged.get(p["arxiv_id"], {}).get("llm_score")
        ordered = rerank_by_actionability(rr_candidates) if args.rr_rerank else rr_candidates
        rr_topn = ordered[:10]
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
            rr_toppicks = _apply_finescale(dest, rr_topn, keys, args)
    else:
        rr_topn = rr_candidates[:10]
        rr_toppicks = [p for p, s in rr_ranked[:10] if s >= TOP_THRESHOLD]
        print(
            f"        RepoRadar: {len(rr_topn)} ranked, "
            f"{len(rr_toppicks)} in Top Picks tier (>=0.5)"
        )

    # 2. Baseline (Opus, via Claude Code CLI or the Anthropic API) -> verify
    b = baseline_mod.run_baseline(
        dest,
        repo_name=name,
        repo_context=repo_context,
        mode=args.baseline,
        mock=args.mock,
        use_cache=not args.no_cache,
    )
    baseline_status = b.get("status", "ok")
    if baseline_status != "ok":
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
        pool.setdefault(p["arxiv_id"].split("v")[0], p)
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
            if (bid := p["arxiv_id"].split("v")[0]) in verdicts
        ]

    pool_gains = [int(v["score"]) for v in verdicts.values()]
    rr_pick_metrics = summarize_system(gains_for(rr_toppicks), pool_gains, n_hallucinated=0)
    rr_topn_metrics = summarize_system(gains_for(rr_topn), pool_gains, n_hallucinated=0)

    if baseline_ok:
        # Restrict to papers still in the pool (a dropped one has no gain).
        b_present = [p for p in b_papers if p["arxiv_id"].split("v")[0] in verdicts]
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
    parser.add_argument("--case", help="Only run this case.")
    parser.add_argument(
        "--mock", action="store_true", help="Mock judge + baseline (no keys/spend)."
    )
    parser.add_argument(
        "--model", default=judge_mod.DEFAULT_JUDGE_MODEL, help="OpenAI judge model."
    )
    parser.add_argument("--sources", default="arxiv", help="RepoRadar sources (comma-separated).")
    parser.add_argument(
        "--baseline",
        choices=["cli", "api"],
        default="cli",
        help="Opus baseline mode: 'cli' = Claude Code headless (needs `claude` on PATH); "
        "'api' = Anthropic Messages API + web_search (needs ANTHROPIC_API_KEY, no CLI).",
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
    for case in bench["cases"]:
        if args.case and case["name"] != args.case:
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
        out = RESULTS_DIR / f"judge-{args.model}-{stamp}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
