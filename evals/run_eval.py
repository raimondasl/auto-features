"""RepoRadar evaluation benchmark runner.

Two modes:

  offline (default) — profile each case's mini-repo and rank a *frozen* pool of
      real arXiv papers (fixtures/<case>.json). Deterministic, no network, no
      API keys. This is the regression benchmark.

  live            — clone each case's real GitHub repo and run the full pipeline
      against real sources (arXiv, and optionally OpenAlex / Semantic Scholar).
      Needs network and, for non-arXiv sources, API keys (see evals/README.md).

Usage:
    uv run python evals/run_eval.py                       # offline, all cases
    uv run python evals/run_eval.py --case rag            # offline, one case
    uv run python evals/run_eval.py --embeddings          # add embedding signal
    uv run python evals/run_eval.py --live                # live, arXiv only
    uv run python evals/run_eval.py --live --sources arxiv,openalex

Keys are read from environment variables, optionally seeded from evals/.env.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402  (needs evals/ on sys.path first)
    EVALS_DIR,
    build_ranking_config,
    load_benchmark,
    profile_case_repo,
    rank_pool,
    resolve_repo_dir,
)
from metrics import evaluate_ranking  # noqa: E402

from reporadar.collector import CollectionError  # noqa: E402

FIXTURES_DIR = EVALS_DIR / "fixtures"
WORK_DIR = EVALS_DIR / ".work"

ENV_KEYS = [
    "OPENALEX_API_KEY",
    "HF_TOKEN",
    "SEMANTIC_SCHOLAR_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
]


# ── environment / keys ────────────────────────────────────────────────────


def load_dotenv(path: Path) -> None:
    """Seed os.environ from a simple KEY=VALUE .env file (existing vars win)."""
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


def collected_keys() -> dict[str, str]:
    return {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}


# ── offline mode ──────────────────────────────────────────────────────────


def load_fixture(name: str) -> list[dict[str, Any]] | None:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def eval_negative_control(ranked: list[dict[str, Any]], case: dict, k: int) -> dict[str, Any]:
    threshold = case.get("max_score_threshold", 0.5)
    scores = [r["score_total"] for r in ranked]
    max_score = max(scores) if scores else 0.0
    top_k = scores[:k]
    n_top_tier = sum(1 for s in scores if s >= threshold)
    return {
        "max_score": max_score,
        "mean_top_k": sum(top_k) / len(top_k) if top_k else 0.0,
        "n_top_tier": n_top_tier,
        "threshold": threshold,
        "passed": max_score < threshold,
        "n_candidates": len(ranked),
    }


def run_offline(bench: dict, only: str | None, k: int, embeddings: bool) -> int:
    print("\n=== RepoRadar offline benchmark ===")
    print(
        f"(k={k}, embeddings={'on' if embeddings else 'off'}, recency weight=0 for determinism)\n"
    )

    agg: list[dict[str, float]] = []
    missing = 0
    for case in bench["cases"]:
        name = case["name"]
        if only and name != only:
            continue

        pool = load_fixture(name)
        if pool is None:
            print(f"  [{name}] no fixture — run build_fixtures.py first. Skipping.")
            missing += 1
            continue

        repo_dir = resolve_repo_dir(case)
        if not Path(repo_dir).is_dir():
            # A case needs BOTH a fixture and a mini-repo to profile against. The skip
            # above only checked the fixture, so building fixtures for the 8 cases that
            # have no `repos/<case>/` turned a clean skip into a crash mid-run, losing
            # the results already computed for the cases that were fine.
            print(f"  [{name}] no mini-repo at {repo_dir} — skipping (fixture exists).")
            missing += 1
            continue
        profile = profile_case_repo(repo_dir)
        ranked = rank_pool(
            profile,
            pool,
            expected_categories=case["expected_categories"],
            repo_dir=repo_dir,
            embeddings=embeddings,
        )

        if case.get("negative_control"):
            r = eval_negative_control(ranked, case, k)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{name}] negative control: {status}")
            print(
                f"        max_score={r['max_score']:.3f} (threshold {r['threshold']}), "
                f"top-tier papers={r['n_top_tier']}, mean_top{k}={r['mean_top_k']:.3f}"
            )
            continue

        labels = {p["arxiv_id"]: int(p["label"]) for p in pool}
        m = evaluate_ranking(ranked, labels, k)
        agg.append(m)
        print(
            f"  [{name}] P@{k}={m['precision@k']:.3f}  R@{k}={m['recall@k']:.3f}  "
            f"nDCG@{k}={m['ndcg@k']:.3f}  MRR={m['mrr']:.3f}  MAP={m['map']:.3f}  "
            f"sep={m['separation']:+.3f}  ({int(m['n_gold'])} gold / {int(m['n_candidates'])})"
        )

    if agg:
        print("\n  --- mean over labeled cases ---")
        keys = ["precision@k", "recall@k", "ndcg@k", "mrr", "map", "separation"]
        means = {key: sum(m[key] for m in agg) / len(agg) for key in keys}
        print(
            f"  P@{k}={means['precision@k']:.3f}  R@{k}={means['recall@k']:.3f}  "
            f"nDCG@{k}={means['ndcg@k']:.3f}  MRR={means['mrr']:.3f}  "
            f"MAP={means['map']:.3f}  sep={means['separation']:+.3f}"
        )
    if missing:
        # Naming the actual cause matters: `build_fixtures.py` cannot create a mini-repo,
        # so telling someone to run it when the fixture already exists sends them in a
        # loop. A case needs a `repos/<case>/` (hand-authored README + manifest) too.
        print(
            f"\n  {missing} case(s) skipped for a missing fixture or mini-repo — see the "
            f"per-case lines above. Fixtures: uv run python evals/build_fixtures.py; "
            f"mini-repos are hand-authored under evals/repos/<case>/."
        )
    return 0


# ── live mode ─────────────────────────────────────────────────────────────


def clone_repo(url: str, dest: Path, *, reuse: bool) -> Path | None:
    if dest.exists():
        if reuse:
            return dest
        print(f"        (reusing existing clone at {dest})")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"        cloning {url} (shallow)...")
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


def collect_live(
    profile: Any,
    case: dict,
    sources: list[str],
    keys: dict[str, str],
    lookback_days: int,
) -> list[dict[str, Any]]:
    """Mirror the live collection path: build queries, fetch from each source."""
    from reporadar.collector import (
        CollectionError,
        build_queries,
        collect_papers,
        to_plain_keywords,
    )
    from reporadar.config import ArxivConfig, QueriesConfig
    from reporadar.paper_id import dedup_id

    categories = case["expected_categories"] or ["cs.LG", "cs.CL", "cs.CV", "cs.SE"]
    arxiv_cfg = ArxivConfig(
        categories=categories, max_results_per_query=50, lookback_days=lookback_days
    )
    queries = build_queries(profile, QueriesConfig(), arxiv_cfg)

    papers: list[dict[str, Any]] = []
    try:
        papers = collect_papers(queries, arxiv_cfg)
    except CollectionError as exc:
        # RAISE, do not degrade — the rule `evals/harness.py` already follows and the one
        # C-4 was paid for. Swallowing this scored a throttled arXiv fetch as an honest
        # result; with a second source enabled it is worse than an empty pool, because
        # the case then runs on the non-arXiv half alone and looks like a measurement.
        print(f"        ! arXiv collection FAILED (not an empty result): {exc}")
        raise

    # Version-insensitive, like cli.update and evals/harness.py. This was the third copy
    # of the C-12 merge and the only one left on raw equality; the guard in
    # tests/test_collector.py checked the harness by name and never looked here.
    seen = {dedup_id(p["arxiv_id"]) for p in papers}
    plain_queries = [to_plain_keywords(q) for q in queries[:5]]

    if "openalex" in sources:
        from reporadar.sources.openalex import collect_papers as oa_collect

        for p in oa_collect(
            plain_queries,
            lookback_days=lookback_days,
            api_key=keys.get("OPENALEX_API_KEY"),
        ):
            if dedup_id(p["arxiv_id"]) not in seen:
                papers.append(p)
                seen.add(dedup_id(p["arxiv_id"]))

    if "semantic_scholar" in sources:
        from reporadar.sources.semantic_scholar import collect_papers as ss_collect

        for p in ss_collect(
            plain_queries,
            api_key=keys.get("SEMANTIC_SCHOLAR_API_KEY"),
            lookback_days=lookback_days,
        ):
            if dedup_id(p["arxiv_id"]) not in seen:
                papers.append(p)
                seen.add(dedup_id(p["arxiv_id"]))

    return papers


def domain_purity(
    ranked: list[dict[str, Any]],
    papers_by_id: dict[str, dict[str, Any]],
    expected: list[str],
    k: int,
) -> float:
    """Fraction of top-k papers whose categories intersect the expected set.

    Score dicts from the ranker don't carry categories, so we look them up from
    the original collected papers by arxiv_id.
    """
    if not expected:
        return float("nan")
    top = ranked[:k]
    if not top:
        return 0.0
    hits = 0
    for r in top:
        cats = papers_by_id.get(r["arxiv_id"], {}).get("categories", [])
        if set(cats) & set(expected):
            hits += 1
    return hits / len(top)


def run_live(
    bench: dict, only: str | None, k: int, embeddings: bool, sources: list[str], reuse: bool
) -> int:
    keys = collected_keys()
    print("\n=== RepoRadar live benchmark ===")
    print(f"(k={k}, sources={','.join(sources)}, embeddings={'on' if embeddings else 'off'})")
    print(f"keys present: {', '.join(keys) or 'none (arXiv only)'}\n")

    if ("openalex" in sources) and "OPENALEX_API_KEY" not in keys:
        print("  ! openalex requested but OPENALEX_API_KEY not set — it will run throttled.\n")

    WORK_DIR.mkdir(exist_ok=True)
    for case in bench["cases"]:
        name = case["name"]
        if only and name != only:
            continue
        print(f"  [{name}] {case['live_repo']}")
        dest = clone_repo(case["live_repo"], WORK_DIR / name, reuse=reuse)
        if dest is None:
            continue

        profile = profile_case_repo(dest)
        try:
            papers = collect_live(profile, case, sources, keys, lookback_days=90)
        except CollectionError as exc:
            # Skipped, and said so. The alternative — scoring the case on whatever the
            # other sources returned — is a domain-purity number for a pool missing its
            # arXiv half, indistinguishable in the output from a real one.
            print(f"        SKIPPED (collection failed, not scored): {exc}\n")
            continue
        if not papers:
            print("        no papers collected.\n")
            continue

        ranked = rank_pool(
            profile,
            papers,
            expected_categories=case["expected_categories"] or ["cs.LG"],
            repo_dir=dest,
            embeddings=embeddings,
            ranking_cfg=build_ranking_config(embeddings=embeddings, w_category=0.5),
        )
        papers_by_id = {p["arxiv_id"]: p for p in papers}
        scores = [r["score_total"] for r in ranked]
        purity = domain_purity(ranked, papers_by_id, case["expected_categories"], k)
        n_top_tier = sum(1 for s in scores if s >= 0.5)

        purity_str = "n/a" if purity != purity else f"{purity:.2f}"  # NaN check
        note = ""
        if case.get("negative_control"):
            thr = case.get("max_score_threshold", 0.5)
            note = f"  [negative control: {'PASS' if max(scores) < thr else 'FAIL'}]"
        print(
            f"        collected={len(papers)}  top_score={max(scores):.3f}  "
            f"mean_top{k}={sum(scores[:k]) / min(k, len(scores)):.3f}  "
            f"domain_purity@{k}={purity_str}  top-tier(>=0.5)={n_top_tier}{note}"
        )
        for r in ranked[:3]:
            title = papers_by_id.get(r["arxiv_id"], {}).get("title", "")
            print(f"          {r['score_total']:.3f}  {r['arxiv_id']}  {title[:70]}")
        print()
    return 0


# ── entry point ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--live", action="store_true", help="Live mode (real repos + sources).")
    parser.add_argument("--case", help="Only run this case.")
    parser.add_argument("--k", type=int, default=None, help="Override the k cutoff.")
    parser.add_argument("--embeddings", action="store_true", help="Add the embedding signal.")
    parser.add_argument(
        "--sources",
        default="arxiv",
        help="Live sources, comma-separated (arxiv,openalex,semantic_scholar).",
    )
    parser.add_argument("--reuse-clones", action="store_true", help="Reuse existing live clones.")
    args = parser.parse_args()

    load_dotenv(EVALS_DIR / ".env")
    bench = load_benchmark()
    k = args.k or bench.get("k", 10)

    if args.live:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        return run_live(bench, args.case, k, args.embeddings, sources, args.reuse_clones)
    return run_offline(bench, args.case, k, args.embeddings)


if __name__ == "__main__":
    raise SystemExit(main())
