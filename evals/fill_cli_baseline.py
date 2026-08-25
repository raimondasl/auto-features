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

**Who pays.** The `cli` baseline authenticates either through a signed-in `claude` CLI, which
bills the user's Claude subscription, or through ANTHROPIC_API_KEY, which bills the API. This
script resolves and *prints* which one a run will use before spending, because the two are
indistinguishable afterwards and may not even give the agent the same tools -- the CLI warns
that a visible key disables connectors. Prefer the subscription:

    claude auth login     # interactive, once; or `claude setup-token` for headless runs

    uv run python evals/fill_cli_baseline.py --dry-run   # $0: what would run, and why
    uv run python evals/fill_cli_baseline.py --compare   # $0: cli vs api where both exist
    uv run python evals/fill_cli_baseline.py             # the cases with no cli baseline yet
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
from diagnose_pool import JUDGE, _judge_stem  # noqa: E402
from harness import WORK_DIR, assemble_repo_context, clone_repo  # noqa: E402
from run_judge_eval import load_dotenv  # noqa: E402
from verify import resolve_references  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

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


def _judged(case: str, ids: list[str]) -> list[int]:
    """Judge scores for *ids* in *case*, skipping any this judge never scored."""
    out = []
    for paper_id in ids:
        for verdict in (JUDGE / case).glob(f"{_judge_stem(paper_id)}*.json"):
            out.append(int(json.loads(verdict.read_text(encoding="utf-8"))["score"]))
            break
    return out


def compare_modes() -> int:
    """$0: `cli` against `api` on every case that has both. Reproduces [P13].

    P13 measured the two modes as different systems on the original 25 and was computed by
    hand; this is the same comparison from the same caches, and it now extends to whichever
    scientific cases have acquired a `cli` run.
    """
    rows: list[dict[str, Any]] = []
    for cache in sorted(CLI_CACHE.glob("*.json")):
        case = cache.stem
        other = EVALS / "cache" / "baseline" / "api" / f"{case}.json"
        if not other.is_file():
            continue
        cli = json.loads(cache.read_text(encoding="utf-8"))
        api = json.loads(other.read_text(encoding="utf-8"))
        if cli.get("status") != "ok" or api.get("status") != "ok":
            continue
        cli_ids = [dedup_id(i) for i in cli.get("ids") or []]
        api_ids = [dedup_id(i) for i in api.get("ids") or []]
        rows.append(
            {
                "case": case,
                "cohort": "scisoft" if case.startswith(("bio-", "mat-")) else "benchmark25",
                "cli": cli_ids,
                "api": api_ids,
                "cli_scores": _judged(case, cli_ids),
                "api_scores": _judged(case, api_ids),
                "shared": sorted(set(cli_ids) & set(api_ids)),
            }
        )

    def summarise(label: str, subset: list[dict[str, Any]]) -> None:
        print(f"\n{label} -- {len(subset)} case(s) with both modes cached")
        print(f"  {'':<6}{'picks':>7}{'judged':>8}{'actionable':>12}{'precision':>11}{'net@2':>9}")
        for mode in ("cli", "api"):
            picks = sum(len(r[mode]) for r in subset)
            scores = [s for r in subset for s in r[f"{mode}_scores"]]
            good = sum(1 for s in scores if s >= ACTIONABLE)
            net = sum(1 if s >= ACTIONABLE else -2 for s in scores) / len(subset)
            prec = f"{good / len(scores):.3f}" if scores else "n/a"
            print(f"  {mode:<6}{picks:>7}{len(scores):>8}{good:>12}{prec:>11}{net:>+9.2f}")
        shared = sum(len(r["shared"]) for r in subset)
        both = [r for r in subset if r["cli"] and r["api"]]
        agreed = [r for r in both if set(r["cli"]) == set(r["api"])]
        print(f"  shared picks: {shared}   cases where both returned papers: {len(both)}")
        print(f"  ...of which the two modes picked the SAME set: {len(agreed)}")

    if not rows:
        print("no case has both a cli and an api baseline cached.")
        return 0
    summarise("ALL", rows)
    for cohort in ("benchmark25", "scisoft"):
        subset = [r for r in rows if r["cohort"] == cohort]
        if subset:
            summarise(cohort, subset)
    print(
        "\nThe two modes are not one baseline under different transport [P13]. Read any "
        "cross-cohort\nclaim with the mode it was measured against, not just the case set."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the `cli` baseline where none exists yet.")
    ap.add_argument("--case", help="Comma-separated subset (still skips cases that have a cache).")
    ap.add_argument("--dry-run", action="store_true", help="$0: list what would run.")
    ap.add_argument("--compare", action="store_true", help="$0: cli vs api where both exist [P13].")
    ap.add_argument("--model", default=judge_mod.DEFAULT_JUDGE_MODEL, help="Judge model.")
    ap.add_argument("--out", help="Write the per-case summary here as JSON.")
    args = ap.parse_args()

    if args.compare:
        return compare_modes()

    load_dotenv(EVALS / ".env")
    if not args.dry_run:
        # The judge is unconditional -- no key, no gold targets, and the baseline spend
        # would be wasted.
        if not os.environ.get("OPENAI_API_KEY"):
            print("! OPENAI_API_KEY is not set (evals/.env); the judge cannot run, so the")
            print("  baseline spend would buy picks nobody can score. Refusing.")
            return 1
        # The baseline needs EITHER a signed-in CLI (billed to the subscription, preferred)
        # or the API key (billed per token). Resolve it here and say which, because the two
        # are not visibly different afterwards and they may not offer the agent the same
        # tools -- the CLI's own warning is that a present key disables connectors.
        auth = baseline_mod.cli_auth_mode()
        if auth == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("! the `claude` CLI is signed out and ANTHROPIC_API_KEY is unset, so the")
            print("  baseline cannot authenticate at all. Either:")
            print("    claude auth login          # bills your Claude subscription")
            print("    claude setup-token         # same, long-lived, for headless runs")
            print("  or put ANTHROPIC_API_KEY in evals/.env to bill the API instead.")
            return 1
        billed = "your Claude subscription" if auth == "subscription" else "the API key"
        print(f"baseline auth: {auth} -- these runs bill {billed}.")
        if auth == "api":
            print(
                f"  (`claude auth login` then re-run to use the subscription; "
                f"{baseline_mod._CLI_AUTH_ENV}=subscription to require it.)"
            )

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
        try:
            row = run_case(case, model=args.model)
        except Exception as exc:  # noqa: BLE001 -- one bad case must not abort the batch
            # 2026-08-25: a UnicodeDecodeError in `subprocess.run`'s reader thread left
            # `stdout` unset, and the TypeError four frames later killed an 11-case run on
            # its first case -- after it had been billed for the answer. The decode bug is
            # fixed; this is the second line of defence, because the next unhandled thing
            # will not be that one.
            print(f"      !! crashed: {type(exc).__name__}: {str(exc)[:200]}")
            row = {"case": case["name"], "status": "crashed", "raw": str(exc)[:200]}
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
    # Under subscription auth the CLI still reports `total_cost_usd`, but it is what these
    # tokens WOULD have cost on the API rather than money spent. Calling that "spend" would
    # be a fabricated figure, so the label follows the auth mode.
    label = "baseline spend" if auth == "api" else "baseline cost-equivalent (subscription)"
    print(f"\n{len(ok)}/{len(rows)} ran. {gold} new gold target(s). {label} ${spent:.2f}")
    if any(r["status"] != "ok" for r in rows):
        print(
            "  NB: failed runs report no cost, so the figure above counts successes only —"
            "\n      a turn-limit failure is a full agentic run that this cannot see."
        )
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
