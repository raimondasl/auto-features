"""Tier B: LLM-judged, abstention-aware actionable-improvement benchmark.

For each real repo, two systems produce a paper list:
  1. RepoRadar  — its "Top Picks" tier (score >= 0.5), which may be empty.
  2. Baseline   — Opus 4.8 via Claude Code headless (the strong baseline).

A neutral OpenAI judge (default GPT-5.5), blind to the source, scores the pooled
union of both lists 0-3 for whether each paper could genuinely improve THIS repo.
Metrics reward precision and correct abstention and penalize false positives.

    # dry-run the whole pipeline with no keys/spend (mock judge + mock baseline):
    uv run python evals/run_judge_eval.py --mock

    # the real thing (needs OPENAI_API_KEY and Claude Code installed):
    uv run python evals/run_judge_eval.py --case rag
    uv run python evals/run_judge_eval.py            # all cases

See evals/README.md for keys and cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
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

from reporadar.config import QueriesConfig, RankingConfig  # noqa: E402
from reporadar.digest import TOP_THRESHOLD  # noqa: E402
from reporadar.ranker import rank_papers  # noqa: E402

RESULTS_DIR = EVALS_DIR / "results"
ENV_KEYS = ["OPENAI_API_KEY", "OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"]
RECENT_DAYS = 180  # baseline papers newer than this count as "recent"


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
    repo_dir: Path, categories: list[str], sources: list[str], keys: dict[str, str], top_n: int = 10
) -> list[tuple[dict[str, Any], float]]:
    """RepoRadar's real ranking: top-N (paper, score) best-first.

    Top Picks (the abstention-respecting output) = those with score >= 0.5.
    Returning the top-N regardless lets us tell a conservative threshold apart
    from genuinely shallow ranking.
    """
    profile = profile_case_repo(repo_dir)
    papers = collect_live_papers(profile, categories, sources=sources, keys=keys, lookback_days=90)
    if not papers:
        return []
    ranking_cfg = RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3)
    ranked = rank_papers(
        papers, profile, ranking_cfg, QueriesConfig(), categories or ["cs.LG"], lookback_days=90
    )
    by_id = {p["arxiv_id"]: p for p in papers}
    out: list[tuple[dict[str, Any], float]] = []
    for s in ranked[:top_n]:
        if s["arxiv_id"] in by_id:
            out.append((by_id[s["arxiv_id"]], s["score_total"]))
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

    # 1. RepoRadar ranking -> Top-10 (diagnostic) and Top Picks (>=0.5, headline)
    rr_ranked = reporadar_ranked(dest, categories, args.sources, keys)
    rr_topn = [p for p, _ in rr_ranked]
    rr_toppicks = [p for p, s in rr_ranked if s >= TOP_THRESHOLD]
    print(f"        RepoRadar: {len(rr_topn)} ranked, {len(rr_toppicks)} in Top Picks tier (>=0.5)")

    # 2. Baseline (Opus via Claude Code) -> verify against real arXiv
    b = baseline_mod.run_baseline(dest, repo_name=name, mock=args.mock, use_cache=not args.no_cache)
    b_papers, n_halluc = resolve_references(b["ids"], b["titles"])
    print(
        f"        Baseline recommended {len(b['ids']) + len(b['titles'])} ref(s) -> "
        f"{len(b_papers)} real, {n_halluc} hallucinated (cost ${b.get('cost_usd', 0):.2f})"
    )

    # 3. Pool = RepoRadar top-N ∪ baseline; judge each once, blind to source
    pool: dict[str, dict[str, Any]] = {}
    for p in rr_topn + b_papers:
        pool.setdefault(p["arxiv_id"].split("v")[0], p)
    verdicts: dict[str, dict[str, Any]] = {}
    for base_id, paper in pool.items():
        verdicts[base_id] = judge_mod.judge_paper(
            name, repo_context, paper, model=args.model, mock=args.mock, use_cache=not args.no_cache
        )

    def gains_for(papers: list[dict[str, Any]]) -> list[int]:
        return [int(verdicts[p["arxiv_id"].split("v")[0]]["score"]) for p in papers]

    pool_gains = [int(v["score"]) for v in verdicts.values()]
    rr_pick_metrics = summarize_system(gains_for(rr_toppicks), pool_gains, n_hallucinated=0)
    rr_topn_metrics = summarize_system(gains_for(rr_topn), pool_gains, n_hallucinated=0)
    b_gains = gains_for(b_papers)
    b_metrics = summarize_system(b_gains, pool_gains, n_hallucinated=n_halluc)

    # Baseline age: recent-only net value (apples-to-apples with RepoRadar).
    recent_gains = [
        g for p, g in zip(b_papers, b_gains, strict=True) if is_recent(p.get("published", ""))
    ]
    b_metrics["n_recent"] = len(recent_gains)
    b_metrics["net_value_recent@2"] = summarize_system(recent_gains, pool_gains)["net_value@2"]

    n_relevant = sum(1 for g in pool_gains if g >= 2)
    print(f"        pool judged: {len(pool_gains)} papers, {n_relevant} genuinely actionable (>=2)")
    _print_system("RepoRadar[TopPicks]", rr_pick_metrics)
    _print_system("RepoRadar[Top10]   ", rr_topn_metrics)
    _print_system(
        "Baseline           ", b_metrics, extra=f"recent={b_metrics['n_recent']}/{len(b_papers)}"
    )

    return {
        "case": name,
        "repo": case["live_repo"],
        "pool_size": len(pool_gains),
        "n_actionable_in_pool": n_relevant,
        "reporadar_toppicks": rr_pick_metrics,
        "reporadar_top10": rr_topn_metrics,
        "baseline": b_metrics,
    }


def _print_system(label: str, m: dict[str, Any], extra: str = "") -> None:
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
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached verdicts/baseline.")
    args = parser.parse_args()
    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    load_dotenv(EVALS_DIR / ".env")
    keys = {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}

    judge_label = "mock" if args.mock else args.model
    baseline_label = "mock" if args.mock else "claude-opus-4-8"
    print("=== RepoRadar Tier B: actionable-improvement benchmark ===")
    print(f"judge={judge_label}  baseline={baseline_label}")
    if not args.mock and "OPENAI_API_KEY" not in keys:
        print("\n! OPENAI_API_KEY not set. Set it (see evals/README.md) or use --mock.")
        return 1

    bench = load_benchmark()
    WORK_DIR.mkdir(exist_ok=True)
    results = []
    for case in bench["cases"]:
        if args.case and case["name"] != args.case:
            continue
        r = run(case, keys, args)
        if r:
            results.append(r)

    if results and not args.mock:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = RESULTS_DIR / f"judge-{args.model}-{stamp}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
