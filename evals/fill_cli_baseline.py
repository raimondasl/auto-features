"""Give the scientific-software cases the same comparator the other 25 were measured against.

Twelve `bio-*` / `mat-*` cases have never had a `cli` baseline run. They have `api` runs, and
until [P13] that looked like the same measurement under a different transport. It is not:
over the 25 shared cases `cli` returns 64 picks at precision 0.889 and +1.68 net@2/case while
`api` returns 34 at 0.824 and +0.64, they share only 10 picks, and there is **no case where
both found papers and agreed**. So those twelve are currently scored against a comparator
roughly 2.6x weaker than the rest of the benchmark, and any cross-case claim that mixes them
is comparing two different systems.

This script runs *only* the two steps a gold set needs -- the `cli` baseline, then the judge
over whatever it recommends -- through the same modules `run_judge_eval` calls, so no prompt
is reimplemented here (the C-3 rule). Both write to the shared caches, so a later full paired
run on these cases reuses this work and pays only for RepoRadar's side.

**What it deliberately will not do.** `diagnose_pool.actionable_baseline_ids` derives the gold
set from the baseline cache, so re-running a baseline for a case that already has one
*redefines ground truth* -- the failure that moved `graph`'s gold set 3 -> 4 on 2026-08-09 and
would have shifted the denominator of every published recall figure. This script therefore
refuses any case that already has a `cli` cache, and refuses to run at all if the cache
discriminator has moved (which would invalidate all 25 existing caches at once). Neither
guard is overridable from the command line; moving the gold set should require editing code
and meaning it.

    uv run python evals/fill_cli_baseline.py --dry-run   # $0: what would run, and why
    uv run python evals/fill_cli_baseline.py             # ~$10: the 12 missing cases
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
import judge as judge_mod  # noqa: E402
from harness import WORK_DIR, assemble_repo_context, clone_repo  # noqa: E402
from run_judge_eval import load_dotenv  # noqa: E402
from verify import resolve_references  # noqa: E402

EVALS = Path(__file__).resolve().parent
CLI_CACHE = EVALS / "cache" / "baseline" / "cli"
ACTIONABLE = 2

# The discriminator every existing `cli` cache was written under. `_discriminator` hashes
# BASELINE_MODEL, BASELINE_PROMPT and the flags, so a change here means the 25 published
# caches would all re-run -- silently redefining the gold set for the whole benchmark.
PINNED_DISCRIMINATOR = "da766b38114e"


def missing_cases(bench: dict[str, Any], only: set[str] | None) -> list[dict[str, Any]]:
    """Cases with a live repo and no `cli` baseline cache yet."""
    out = []
    for case in bench["cases"]:
        name = case["name"]
        if only is not None and name not in only:
            continue
        if not case.get("live_repo"):
            continue
        if (CLI_CACHE / f"{name}.json").is_file():
            continue
        out.append(case)
    return out


def run_case(case: dict[str, Any], *, model: str) -> dict[str, Any]:
    name = case["name"]
    dest = clone_repo(case["live_repo"], WORK_DIR / name, reuse=True)
    if dest is None:
        return {"case": name, "status": "clone_failed"}
    repo_context = assemble_repo_context(dest)

    result = baseline_mod.run_baseline(dest, repo_name=name, repo_context=repo_context, mode="cli")
    status = result.get("status", "ok")
    if status != "ok":
        # Failures are never cached, so this retries next run rather than poisoning the set.
        return {"case": name, "status": status, "raw": (result.get("raw") or "")[:200]}

    papers, hallucinated, lookup_failed = resolve_references(result["ids"], result["titles"])
    if lookup_failed:
        # An arXiv outage is not the baseline's fault, and a paper we could not verify must
        # not be judged as if we had (C-15's rule: a collection failure is loud, not scored).
        return {
            "case": name,
            "status": "arxiv_unverified",
            "n_lookup_failed": lookup_failed,
            "cost_usd": result.get("cost_usd", 0.0),
        }

    scores: dict[str, int] = {}
    failed = 0
    for paper in papers:
        try:
            verdict = judge_mod.judge_paper(name, repo_context, paper, model=model)
        except Exception as exc:  # noqa: BLE001 -- never score an unjudged paper as 0
            failed += 1
            print(f"      ! judge failed for {paper['arxiv_id']}: {str(exc)[:120]}")
            continue
        scores[paper["arxiv_id"]] = int(verdict["score"])

    gold = sorted(pid for pid, s in scores.items() if s >= ACTIONABLE)
    return {
        "case": name,
        "status": "ok",
        "n_returned": len(result["ids"]) + len(result["titles"]),
        "n_resolved": len(papers),
        "n_hallucinated": hallucinated,
        "n_judged": len(scores),
        "n_judge_failed": failed,
        "gold_targets": gold,
        "net_at_2": sum(1 if s >= ACTIONABLE else -2 for s in scores.values()),
        "cost_usd": result.get("cost_usd", 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the `cli` baseline where none exists yet.")
    ap.add_argument("--case", help="Comma-separated subset (still skips cases that have a cache).")
    ap.add_argument("--dry-run", action="store_true", help="$0: list what would run.")
    ap.add_argument("--model", default=judge_mod.DEFAULT_JUDGE_MODEL, help="Judge model.")
    ap.add_argument("--out", help="Write the per-case summary here as JSON.")
    args = ap.parse_args()

    # The `claude` subprocess inherits this process's environment, and without the key it
    # exits 1 with "Not logged in" -- which `run_baseline` correctly reports as an error
    # rather than an abstention, but only after paying for the round trip.
    load_dotenv(EVALS / ".env")
    if not args.dry_run:
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            if not os.environ.get(key):
                print(f"! {key} is not set (evals/.env). The baseline needs the first and")
                print("  the judge the second; running without either wastes the other.")
                return 1

    live = baseline_mod._discriminator("cli", "", None)
    if live != PINNED_DISCRIMINATOR:
        print(f"! the `cli` cache discriminator moved: {PINNED_DISCRIMINATOR} -> {live}")
        print("  Every existing cli cache would re-run, redefining the gold set for all 25")
        print("  published cases. Refusing. If that is genuinely intended, update the pin.")
        return 1

    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    only = set(args.case.split(",")) if args.case else None
    if only is not None:
        known = {c["name"] for c in bench["cases"]}
        if unknown := only - known:
            print(f"! unknown case(s): {sorted(unknown)}")
            return 1
    todo = missing_cases(bench, only)
    have = sorted(p.stem for p in CLI_CACHE.glob("*.json"))

    print(f"cli baselines on disk: {len(have)}")
    print(f"cases with no cli baseline: {len(todo)}")
    for case in todo:
        print(f"  {case['name']:<18} {case['live_repo']}")
    if only and (skipped := sorted(only - {c["name"] for c in todo})):
        print("\nskipped (already have a cli cache -- re-running would move the gold set):")
        for name in skipped:
            print(f"  {name}")
    if args.dry_run or not todo:
        print("\n(dry run)" if args.dry_run else "\nnothing to do.")
        return 0

    rows = []
    spent = 0.0
    for i, case in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {case['name']}")
        row = run_case(case, model=args.model)
        rows.append(row)
        spent += float(row.get("cost_usd") or 0.0)
        if row["status"] != "ok":
            print(f"      !! did not run [{row['status']}] {row.get('raw', '')}")
            continue
        print(
            f"      {row['n_returned']} recommended -> {row['n_resolved']} real "
            f"({row['n_hallucinated']} hallucinated), {row['n_judged']} judged, "
            f"{len(row['gold_targets'])} actionable, net@2 {row['net_at_2']:+.0f} "
            f"(${row['cost_usd']:.2f})"
        )

    ok = [r for r in rows if r["status"] == "ok"]
    gold = sum(len(r["gold_targets"]) for r in ok)
    print(f"\n{len(ok)}/{len(rows)} ran. {gold} new gold target(s). baseline spend ${spent:.2f}")
    if ok:
        mean = sum(r["net_at_2"] for r in ok) / len(ok)
        print(f"baseline net@2 over the cases that ran: {mean:+.2f}/case")
    print("\nNext: `uv run python evals/freeze_gold_targets.py` to re-freeze, and read the diff")
    print("      -- these targets ADD to the frozen set; nothing existing should move.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
