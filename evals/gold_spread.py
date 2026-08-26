"""How much of the gold set is a property of the draw? k independent redraws. [P17]

Every published recall denominator -- 21/56, 34/56, 43/56 -- divides by a gold set derived
from **one** run of the agentic baseline. P15 measured what a re-run costs at the PICK level:
a fresh draw of the identical configuration disagrees with the stored answer on ~59% of picks
(mean Jaccard 0.41). What that leaves open is the question the denominators actually depend
on, because picks are not targets -- the judge filters them, and it may absorb most of the
churn or none of it.

This runs the baseline **k = 3 more times over the 25 benchmark cases**, at unchanged flags,
and judges every pick. Two different questions come out, and they must not be conflated:

* **Reproducibility of the frozen set.** For each of the 56 frozen targets, in how many of
  the k fresh draws does it reappear? This is the one that prices the published `/56`.
* **Spread among the fresh draws.** Pairwise agreement between draws that share a session,
  an auth path and a date -- the noise floor with the stored cache's confounds removed.

**Pre-registered, before the first call:**

* **Prediction.** Target-level reproducibility should EXCEED the pick-level 0.41, because the
  judge filter is a stable function applied to a noisy input: a draw that finds a different
  paper on the same topic still yields a target if the judge scores it >= 2. If target-level
  agreement is not meaningfully above 0.41, the judge absorbs nothing and the gold set is as
  noisy as the search.
* **Decision rule.** If a fresh draw reproduces **< 2/3** of the frozen targets, the
  denominator is not a stable quantity and every published recall figure needs an interval
  rather than a point. Between 2/3 and 90%, the point figure stands with the spread quoted
  beside it. Above 90%, the draw is not a material source of error.
* **Kill condition.** If more than 3 of 25 cases fail to produce an `ok` baseline in a given
  draw, that draw is incomplete and is reported as such rather than averaged in -- a missing
  case is not a case that found nothing (void, not null, the failure this project keeps
  paying for).

**Nothing touches the shared caches.** Every baseline runs with `use_cache=False`, so the 34
stored answers are neither read nor overwritten. Judge verdicts DO use the shared cache, on
purpose: a verdict is a function of (case, paper, rubric) and is the same object whoever asks
for it, which is also why repeated picks across draws cost nothing.

**Persistence is incremental and resumable.** The artifact is rewritten after every single
case, and a re-invocation skips (draw, case) pairs it already holds. A 75-run job that loses
three hours of subscription usage to one exception is the C-29 shape, and this is a longer
job than the one that taught it.

    uv run python evals/gold_spread.py --dry-run   # $0, the plan and what is already done
    uv run python evals/gold_spread.py             # ~75 agentic runs + judging, resumable
    uv run python evals/gold_spread.py --report    # $0, re-read the artifact
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
from diagnose_pool import ACTIONABLE, JUDGE, _judge_stem  # noqa: E402
from harness import WORK_DIR, assemble_repo_context, clone_repo  # noqa: E402
from run_judge_eval import load_dotenv  # noqa: E402
from verify import resolve_references  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
OUT = EVALS / "gold_spread.json"
GOLD = EVALS / "gold_targets.json"

DRAWS = 3
# The cohort every published denominator is over. The scientific cases were added later and
# no published figure includes them; mixing them in would answer a question nobody asked.
COHORT = "benchmark25"
MAX_FAILED_PER_DRAW = 3  # kill condition


def cohort_cases(bench: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        c
        for c in bench["cases"]
        if c.get("live_repo") and not c["name"].startswith(("bio-", "mat-"))
    ]


def _judge_cached(case: str, paper_id: str) -> int | None:
    for verdict in (JUDGE / case).glob(f"{_judge_stem(paper_id)}*.json"):
        return int(json.loads(verdict.read_text(encoding="utf-8"))["score"])
    return None


def run_one(case: dict[str, Any], *, model: str) -> dict[str, Any]:
    """One fresh draw of one case: baseline (uncached) then judge (cached)."""
    name = case["name"]
    dest = clone_repo(case["live_repo"], WORK_DIR / name, reuse=True)
    if dest is None:
        return {"status": "clone_failed"}
    context = assemble_repo_context(dest)

    result = baseline_mod.run_baseline(
        dest,
        repo_name=name,
        repo_context=context,
        mode="cli",
        use_cache=False,  # never read, never write -- the stored gold set is untouchable here
    )
    if result.get("status") != "ok":
        return {"status": result.get("status", "error"), "raw": (result.get("raw") or "")[:160]}

    picks = [dedup_id(i) for i in result.get("ids") or []]
    papers, hallucinated, lookup_failed = resolve_references(result["ids"], result["titles"])
    scores: dict[str, int] = {}
    judge_failed = 0
    for paper in papers:
        pid = dedup_id(paper["arxiv_id"])
        try:
            scores[pid] = int(judge_mod.judge_paper(name, context, paper, model=model)["score"])
        except Exception as exc:  # noqa: BLE001 -- an unjudged paper is never scored 0
            judge_failed += 1
            print(f"        ! judge failed for {pid}: {str(exc)[:100]}")
    return {
        # `partial` when something could not be verified or judged: the draw's picks are
        # real, but its TARGET set is a floor rather than a count.
        "status": "partial" if (lookup_failed or judge_failed) else "ok",
        "picks": picks,
        "targets": sorted(p for p, s in scores.items() if s >= ACTIONABLE),
        "scores": dict(sorted(scores.items())),
        "n_hallucinated": hallucinated,
        "n_lookup_failed": lookup_failed,
        "n_judge_failed": judge_failed,
        "num_turns": result.get("num_turns"),
        "cost_usd": result.get("cost_usd", 0.0),
    }


def load_artifact() -> dict[str, Any]:
    if OUT.is_file():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {
        "_comment": (
            "k independent redraws of the cli baseline over the benchmark25 cohort, judged, "
            "to price how much of the gold set is a property of the draw. Derived by "
            "evals/gold_spread.py; pinned by tests/test_gold_spread.py. Written incrementally "
            "-- a partial artifact is expected and resumable."
        ),
        "cohort": COHORT,
        "draws": DRAWS,
        "results": {},
    }


def save(artifact: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8")


# ── analysis ───────────────────────────────────────────────────────────────


def _draw_targets(artifact: dict[str, Any], draw: int) -> dict[str, set[str]] | None:
    """{case: targets} for one draw -- **`ok` rows only**.

    A `partial` row is one where arXiv could not resolve a pick or the judge could not score
    one, so its target set is a FLOOR, not a count. Reading it as a count would bias every
    fresh draw downward and manufacture exactly the instability this probe exists to measure:
    the arXiv 429s of 2026-08-26 would have been reported as the gold set moving. Partial
    rows are kept in the artifact and excluded from the arithmetic, and `report` says how
    many were dropped.
    """
    out: dict[str, set[str]] = {}
    for key, row in artifact["results"].items():
        d, case = key.split("/", 1)
        if int(d) == draw and row["status"] == "ok":
            out[case] = set(row.get("targets") or [])
    return out or None


def report(artifact: dict[str, Any]) -> int:
    frozen_all = json.loads(GOLD.read_text(encoding="utf-8"))["targets"]
    frozen = {
        c: {dedup_id(i) for i in ids}
        for c, ids in frozen_all.items()
        if not c.startswith(("bio-", "mat-"))
    }
    n_frozen = sum(len(v) for v in frozen.values())

    ok_by_draw: dict[int, list[str]] = {}
    partial_by_draw: dict[int, list[str]] = {}
    failed_by_draw: dict[int, list[str]] = {}
    for key, row in artifact["results"].items():
        d, case = key.split("/", 1)
        bucket = (
            ok_by_draw
            if row["status"] == "ok"
            else partial_by_draw
            if row["status"] == "partial"
            else failed_by_draw
        )
        bucket.setdefault(int(d), []).append(case)

    print(f"frozen {COHORT} gold set: {n_frozen} targets / {len(frozen)} cases")
    all_draws = sorted(set(ok_by_draw) | set(partial_by_draw) | set(failed_by_draw))
    for d in all_draws:
        good = ok_by_draw.get(d, [])
        part = partial_by_draw.get(d, [])
        bad = failed_by_draw.get(d, [])
        n = len(good) + len(part) + len(bad)
        # The baseline failing to complete is not the baseline finding nothing. It is its own
        # source of denominator movement, so it is reported as a rate rather than hidden.
        flag = (
            "  [INCOMPLETE — figures below are over the cases it covered]"
            if len(bad) > MAX_FAILED_PER_DRAW
            else ""
        )
        print(
            f"  draw {d}: {len(good)} ok, {len(part)} partial (excluded from counts), "
            f"{len(bad)} failed = {len(bad) / n:.0%} failure rate{flag}"
        )
        if bad:
            print(f"           failed: {sorted(bad)}")
        if part:
            print(f"           partial: {sorted(part)}")

    draws = {d: t for d in range(1, DRAWS + 1) if (t := _draw_targets(artifact, d))}
    if not draws:
        print("\nno complete draws yet.")
        return 0

    # --- the number the published denominators depend on -------------------------
    print("\nreproducibility of the frozen targets, per draw (shared cases only):")
    per_draw_repro = []
    for d, targets in sorted(draws.items()):
        shared = set(targets) & set(frozen)
        hit = sum(len(frozen[c] & targets[c]) for c in shared)
        tot = sum(len(frozen[c]) for c in shared)
        if not tot:
            continue
        per_draw_repro.append(hit / tot)
        print(f"  draw {d}: {hit}/{tot} = {hit / tot:.2f}   over {len(shared)} case(s)")

    # --- how big is each draw's own gold set? ------------------------------------
    print("\ngold-set size per draw (the denominator, had we used that draw):")
    for d, targets in sorted(draws.items()):
        shared = set(targets) & set(frozen)
        print(
            f"  draw {d}: {sum(len(targets[c]) for c in shared)} targets"
            f"  vs frozen {sum(len(frozen[c]) for c in shared)} on the same {len(shared)} case(s)"
        )

    # --- union growth: does the target set saturate? -----------------------------
    common = set.intersection(*[set(t) for t in draws.values()]) & set(frozen)
    if common:
        union: set[tuple[str, str]] = set()
        print(f"\nunion growth over {len(common)} case(s) present in every draw:")
        union |= {(c, i) for c in common for i in frozen[c]}
        print(f"  frozen alone:          {len(union)}")
        for d, targets in sorted(draws.items()):
            union |= {(c, i) for c in common for i in targets[c]}
            print(f"  + draw {d}:              {len(union)}")

        # Chao1 at TARGET level (P16's was pick-level), counting the frozen set as an
        # occasion alongside the fresh draws.
        counts: dict[tuple[str, str], int] = {}
        for source in [frozen, *draws.values()]:
            for c in common:
                for i in source.get(c, set()):
                    counts[(c, i)] = counts.get((c, i), 0) + 1
        f1 = sum(1 for v in counts.values() if v == 1)
        f2 = sum(1 for v in counts.values() if v == 2)
        s_obs = len(counts)
        chao1 = s_obs + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
        print(
            f"  Chao1 (target level, {1 + len(draws)} occasions): "
            f"S_obs={s_obs} f1={f1} f2={f2} -> >= {chao1:.1f}"
        )

    if per_draw_repro:
        mean = sum(per_draw_repro) / len(per_draw_repro)
        print(f"\nmean reproducibility of the frozen set: {mean:.2f}")
        print("  pick-level agreement measured by P15 for comparison: 0.41")
        if mean < 2 / 3:
            verdict = "the denominator is NOT stable — published recall needs an interval"
        elif mean < 0.90:
            verdict = "the point figure stands, with this spread quoted beside it"
        else:
            verdict = "the draw is not a material source of error"
        print(f"  VERDICT (pre-registered): {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="k independent redraws of the gold set.")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--case", help="Comma-separated subset (for smoke tests).")
    ap.add_argument("--dry-run", action="store_true", help="$0: plan and progress.")
    ap.add_argument("--report", action="store_true", help="$0: analyse the artifact.")
    ap.add_argument("--model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()

    artifact = load_artifact()
    if args.report:
        return report(artifact)

    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    cases = cohort_cases(bench)
    if args.case:
        want = set(args.case.split(","))
        cases = [c for c in cases if c["name"] in want]

    todo = [
        (d, c)
        for d in range(1, args.draws + 1)
        for c in cases
        if f"{d}/{c['name']}" not in artifact["results"]
    ]
    print(f"cohort: {len(cases)} case(s) x {args.draws} draw(s) = {len(cases) * args.draws} runs")
    print(f"already recorded: {len(artifact['results'])}   remaining: {len(todo)}")
    if args.dry_run:
        print("\n(dry run)")
        return 0
    if not todo:
        print("\nnothing to do; use --report.")
        return report(artifact)

    load_dotenv(EVALS / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        # Picks without verdicts are not targets, and this whole probe is about targets.
        print("! OPENAI_API_KEY is not set; picks would be unjudgeable. Refusing.")
        return 1
    auth = baseline_mod.cli_auth_mode()
    print(f"baseline auth: {auth}   judge: {args.model}   baseline caches NOT touched")

    for n, (draw, case) in enumerate(todo, start=1):
        print(f"\n[{n}/{len(todo)}] draw {draw} / {case['name']}")
        try:
            row = run_one(case, model=args.model)
        except Exception as exc:  # noqa: BLE001 -- one bad case must not end a 75-run job
            print(f"        !! crashed: {type(exc).__name__}: {str(exc)[:160]}")
            row = {"status": "crashed", "raw": str(exc)[:160]}
        artifact["results"][f"{draw}/{case['name']}"] = row
        save(artifact)  # after EVERY case: this job is too long to lose (C-29)
        if row["status"] in ("ok", "partial"):
            print(
                f"        {len(row['picks'])} pick(s), {len(row['targets'])} target(s)"
                f"  turns={row['num_turns']}  [{row['status']}]"
            )
        else:
            print(f"        !! {row['status']}: {row.get('raw', '')[:120]}")

    print(f"\nwrote {OUT.name}\n")
    return report(artifact)


if __name__ == "__main__":
    raise SystemExit(main())
