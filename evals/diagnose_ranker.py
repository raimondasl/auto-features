"""Does the ranker's ordering predict actionability? The one stage never measured.

Every label the benchmark owns comes from the ranker's own top-10 plus the baseline's picks —
a sample of exactly what the ranker already liked. So the ranker has never been scored, the
32% base rate behind the triage numbers is measured on a skewed distribution, and an earlier
attempt to infer ranker quality from the Tier B pool produced an artifact (the pool mixes
RepoRadar's 40% with Opus's 100%; see RESULTS.md).

This judges a **rank-stratified sample** of the candidate pool: papers from ranks 1-10,
11-50, 51-150 and 151+. If the ranker discriminates, the actionable rate falls monotonically
across the strata. If it is flat, position in the ranking carries no information about
usefulness — which would mean the heuristic weights are decoration, and would reframe the
citation-hop work (18/24 recall inside a 92,014-paper pool) as blocked on ranking rather than
on retrieval.

    uv run python evals/diagnose_ranker.py --per-stratum 4      # ~$4-7 of GPT-5.5

**Costs money** (judge calls). Verdicts land in the same `cache/judge/` tree the rest of the
benchmark uses, so they are reused by every later run and by `diagnose_triage.py` — this
enlarges the shared labelled set rather than building a private one.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import judge as judge_mod  # noqa: E402
from harness import (  # noqa: E402
    assemble_repo_context,
    collect_live_papers,
    profile_case_repo,
    rank_pool,
)

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
ACTIONABLE = 2
STRATA = ((1, 10), (11, 50), (51, 150), (151, 10**9))


def _load_env() -> None:
    env = EVALS / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-stratum", type=int, default=4, help="papers judged per rank band")
    ap.add_argument("--case", help="only this case")
    ap.add_argument("--seed", type=int, default=20260802, help="sampling seed, for reproducibility")
    ap.add_argument("--dry-run", action="store_true", help="show the sample, make no judge calls")
    args = ap.parse_args()

    _load_env()
    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rows: list[dict] = []

    for case in bench["cases"]:
        name = case["name"]
        if args.case and name != args.case:
            continue
        repo = WORK / name
        if not repo.is_dir():
            print(f"  [{name}] no clone — skipping")
            continue

        profile = profile_case_repo(repo)
        papers = collect_live_papers(
            profile,
            case["expected_categories"] or ["cs.LG"],
            sources=["arxiv"],
            keys={},
            lookback_days=3650,
            sort_by="relevance",
        )
        if not papers:
            print(f"  [{name}] empty pool — skipping")
            continue
        ranked = rank_pool(
            profile,
            papers,
            expected_categories=case["expected_categories"] or [],
            repo_dir=repo,
            embeddings=False,
        )
        # The judge must see the repo exactly as a Tier B run shows it, or these labels
        # are not interchangeable with the 428 already in the cache.
        repo_context = assemble_repo_context(repo)
        by_id = {p["arxiv_id"]: p for p in papers}
        order = [s["arxiv_id"] for s in ranked]
        print(f"[{name}] pool={len(order)}", flush=True)

        for lo, hi in STRATA:
            band = order[lo - 1 : hi]
            if not band:
                continue
            pick = rng.sample(band, min(args.per_stratum, len(band)))
            for pid in pick:
                paper = by_id.get(pid)
                if not paper:
                    continue
                rank = order.index(pid) + 1
                if args.dry_run:
                    print(f"    rank {rank:5d}  {paper['title'][:64]}")
                    rows.append({"case": name, "id": pid, "rank": rank, "score": None})
                    continue
                try:
                    v = judge_mod.judge_paper(name, repo_context, paper)
                except Exception as exc:  # noqa: BLE001
                    print(f"    rank {rank:5d}  judge failed: {exc}")
                    continue
                s = int(v["score"])
                rows.append({"case": name, "id": pid, "rank": rank, "score": s})
                print(f"    rank {rank:5d}  score={s}  {paper['title'][:56]}", flush=True)

    print("\n=== ACTIONABLE RATE BY RANK BAND ===")
    print(f"{'band':>12} {'n':>4} {'actionable':>11}")
    scored = [r for r in rows if r["score"] is not None]
    for lo, hi in STRATA:
        sub = [r for r in scored if lo <= r["rank"] <= hi]
        if not sub:
            continue
        act = sum(1 for r in sub if r["score"] >= ACTIONABLE)
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"{label:>12} {len(sub):4d} {act / len(sub):10.0%}")
    if scored:
        top = [r for r in scored if r["rank"] <= 10]
        rest = [r for r in scored if r["rank"] > 10]
        if top and rest:
            a = sum(1 for r in top if r["score"] >= ACTIONABLE) / len(top)
            b = sum(1 for r in rest if r["score"] >= ACTIONABLE) / len(rest)
            print(f"\ntop-10 {a:.0%} vs deeper {b:.0%}  ->  lift {a - b:+.0%}")
            print(
                "A ranker with no signal shows no lift; the ordering would then be "
                "decoration and the digest a random draw from the pool."
            )
    (WORK / "diag_ranker.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
