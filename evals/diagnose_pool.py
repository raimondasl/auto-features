"""Is RepoRadar's failure a POOL problem or a SELECTION problem?

Free, keyless, no LLM calls — only arXiv fetches. Run this before any retrieval work.

For every paper the Opus baseline recommended AND the GPT-5.5 judge scored >= 2 (genuinely
actionable), ask one question: did RepoRadar's own queries fetch it at all?

    fetched but not returned  -> SELECTION problem (ranking or triage dropped a good paper)
    never fetched             -> POOL problem (the queries cannot reach it)

Result on 2026-08-01: **0 of 24 reachable across 9 repos, out of 2030 papers fetched.**
Ruled out at the same time: not an id-matching artifact; 23 of the 24 are inside the arXiv
categories being searched; and a precise phrase query returns them at rank 1. See
`RESULTS.md` -> "Candidate-pool diagnosis".

    uv run python evals/diagnose_pool.py

Reads cached baseline responses and judge verdicts, so it re-derives the numbers without
paying for either. It does hit arXiv once per query per case — arXiv IP-blocks on sustained
request rate, so do not loop this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import collect_live_papers, profile_case_repo  # noqa: E402

EVALS = Path(__file__).resolve().parent
BASELINE = EVALS / "cache" / "baseline" / "cli"
JUDGE = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
WORK = EVALS / ".work"
ACTIONABLE = 2


def actionable_baseline_ids(case: str) -> list[str]:
    """arXiv ids the baseline recommended *and* the judge confirmed actionable."""
    f = BASELINE / f"{case}.json"
    if not f.is_file():
        return []
    data = json.loads(f.read_text(encoding="utf-8"))
    if data.get("status") != "ok":
        return []
    out: list[str] = []
    for paper_id in data.get("ids") or []:
        for verdict in (JUDGE / case).glob(f"{paper_id}*.json"):
            if json.loads(verdict.read_text(encoding="utf-8"))["score"] >= ACTIONABLE:
                out.append(paper_id)
            break
    return out


def main() -> int:
    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    rows = []
    for case in bench["cases"]:
        name = case["name"]
        wanted = actionable_baseline_ids(name)
        if not wanted:
            continue  # baseline abstained, or nothing it picked was judged actionable
        repo = WORK / name
        if not repo.is_dir():
            print(f"  [{name}] no clone at {repo}; skipping")
            continue

        profile = profile_case_repo(repo)
        # The benchmark's own discovery config: all-time, relevance-sorted. Using the
        # 14-day default would confound a pool problem with the lookback window, since
        # every target is >= 11 months old.
        papers = collect_live_papers(
            profile,
            case["expected_categories"] or ["cs.LG"],
            sources=["arxiv"],
            keys={},
            lookback_days=3650,
            sort_by="relevance",
        )
        pool = {p["arxiv_id"].split("v")[0] for p in papers}
        hit = [i for i in wanted if i in pool]
        miss = [i for i in wanted if i not in pool]
        rows.append({"case": name, "pool": len(pool), "wanted": wanted, "hit": hit, "miss": miss})
        print(
            f"  [{name}] pool={len(pool):4d}  baseline-good={len(wanted)}  "
            f"in-pool={len(hit)}  NOT-fetched={len(miss)}"
        )
        for i in miss:
            print(f"       missed: {i}")

    want = sum(len(r["wanted"]) for r in rows)
    hit = sum(len(r["hit"]) for r in rows)
    pool = sum(r["pool"] for r in rows)
    print("\n=== VERDICT ===")
    print(f"papers fetched across {len(rows)} cases: {pool}")
    print(f"known-good papers: {want}")
    print(f"  reachable (in pool): {hit}  ({hit / max(want, 1):.0%})")
    print(f"  never fetched:       {want - hit}  ({(want - hit) / max(want, 1):.0%})")
    print()
    if hit / max(want, 1) >= 0.5:
        print("-> Mostly a SELECTION problem: the papers are fetched and then dropped.")
    else:
        print("-> Mostly a POOL problem: the queries never reach these papers.")
        print("   No reranker can recover a paper that was never fetched.")
    (WORK / "diag_pool.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
