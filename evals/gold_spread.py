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
row, under a lock, and a re-invocation skips what it already holds. A 75-run job that loses
three hours of subscription usage to one exception is the C-29 shape, and this is a longer
job than the one that taught it.

**Two phases, because only one of them is rate-limited.** Phase A runs the agentic baselines
and touches nothing we throttle -- `run_baseline` shells out to `claude` and parses the reply.
Phase B verifies against arXiv and judges. The first version interleaved them, which made the
whole job look arXiv-bound when seconds of it were: measured 2026-08-26, four runs at
concurrency 4 compressed 711 s of phase-A work into 275 s of wall clock, with phase B still
strictly serial and no throttling. `--concurrency` therefore applies to phase A only, by
construction rather than by convention.

**The turn cap is a parameter, and draws at different caps are different configurations.**
`--max-turns` is recorded per row and `report` refuses to average across caps -- it analyses
the shipped cap and lists the others separately. Draws are discovered from the artifact, not
assumed to be 1..k, so a trial at a different cap cannot be silently omitted from the figures.
None of this touches `cache/baseline/cli/` or `_discriminator`: the published comparator is a
separate question from the witness generator, and this script only ever does the latter.

**The prompt is a parameter too, and it is a stronger one than the cap.** `--prompt-version
v2` draws with `BASELINE_PROMPT_V2`, which allows non-arXiv papers. That is not a redraw of
this probe's configuration — it is a *different searcher*, so it writes its own artifact
(`gold_spread_v2.json`), its draw numbers cannot collide with these, and `report` refuses to
call its overlap with the frozen set "reproducibility" or to apply the pre-registered
decision rule to it. Those draws exist to widen the witness set (P16), which is the
baseline's witness-generator role and owes no validation; the published comparator is a
separate question and `cache/baseline/cli/` is still never touched here.

    uv run python evals/gold_spread.py --dry-run   # $0, the plan and what is already done
    uv run python evals/gold_spread.py             # k draws at the shipped cap, resumable
    uv run python evals/gold_spread.py --max-turns 30 --concurrency 4    # a faster variant
    uv run python evals/gold_spread.py --report    # $0, re-read the artifact
    uv run python evals/gold_spread.py --prompt-version v2 --max-turns 30 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from reporadar.paper_id import canonical_ref, dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
OUT = EVALS / "gold_spread.json"
GOLD = EVALS / "gold_targets.json"


def out_path(prompt_version: str) -> Path:
    """One artifact per prompt version, rather than one artifact with a version column.

    Rows are keyed `{draw}/{case}`, and every analysis in `report` splits that key. A v2 draw
    sharing the file would collide with v1's draw 1 on the key -- so `todo_baseline` would
    find the work already done and run nothing at all, silently. Separate files make a v2
    draw impossible to mistake for a v1 one, which is the same rule `report` already applies
    to turn caps: different configuration, different figures, never averaged.
    """
    if prompt_version == baseline_mod.DEFAULT_PROMPT_VERSION:
        return OUT
    return EVALS / f"gold_spread_{prompt_version}.json"


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


def _turn_flags(max_turns: int) -> list[str] | None:
    """`CLAUDE_FLAGS` with the cap replaced, or None to use the shipped list unchanged.

    Built from the shipped list rather than retyped, so a change to the model or the allowed
    tools cannot silently diverge between the probe and the product (the C-3 rule).
    """
    if max_turns == baseline_mod.DEFAULT_MAX_TURNS:
        return None
    flags = list(baseline_mod.CLAUDE_FLAGS)
    flags[flags.index("--max-turns") + 1] = str(max_turns)
    return flags


def run_baseline_only(
    case: dict[str, Any],
    *,
    max_turns: int,
    prompt_version: str = baseline_mod.DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    """Phase A: the agentic run alone. **Touches no network service we rate-limit.**

    This is what makes concurrency safe. `run_baseline` shells out to `claude` and parses the
    reply; arXiv and the judge are not involved until phase B. The earlier serial design
    interleaved them, which made the whole job look arXiv-bound when only seconds of it were.
    """
    name = case["name"]
    dest = clone_repo(case["live_repo"], WORK_DIR / name, reuse=True)
    if dest is None:
        return {"status": "clone_failed", "phase": "judged"}
    started = time.monotonic()
    result = baseline_mod.run_baseline(
        dest,
        repo_name=name,
        repo_context=assemble_repo_context(dest),
        mode="cli",
        use_cache=False,  # never read, never write -- the stored gold set is untouchable here
        flags=_turn_flags(max_turns),
        prompt_version=prompt_version,
    )
    elapsed = round(time.monotonic() - started, 1)
    if result.get("status") != "ok":
        # Nothing to judge, so this row is finished; `phase` says so and resume skips it.
        return {
            "status": result.get("status", "error"),
            "phase": "judged",
            "raw": (result.get("raw") or "")[:160],
            "max_turns": max_turns,
            "prompt_version": prompt_version,
            "duration_s": elapsed,
        }
    return {
        "status": "baseline_ok",
        "phase": "baseline",
        # `canonical_ref`, not `dedup_id`: under v2 a pick can be a DOI, and the resolver
        # hands the same paper back in the prefixed `doi:` form. Two spellings of one id
        # would break this artifact's own `targets <= picks` invariant for every non-arXiv
        # paper. For an arXiv id the two functions are identical, so no stored row moves.
        "picks": [canonical_ref(i) for i in result.get("ids") or []],
        "raw_ids": list(result.get("ids") or []),
        "raw_titles": list(result.get("titles") or []),
        "num_turns": result.get("num_turns"),
        "cost_usd": result.get("cost_usd", 0.0),
        "max_turns": max_turns,
        "prompt_version": prompt_version,
        "duration_s": elapsed,
    }


def judge_row(case_name: str, row: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Phase B: verify against arXiv and judge. **Serial, because both are rate-limited.**"""
    dest = WORK_DIR / case_name
    context = assemble_repo_context(dest)
    papers, hallucinated, lookup_failed, unjudgeable = resolve_references(
        row.get("raw_ids") or [], row.get("raw_titles") or []
    )
    scores: dict[str, int] = {}
    judge_failed = 0
    for paper in papers:
        pid = canonical_ref(paper["arxiv_id"])
        try:
            scores[pid] = int(
                judge_mod.judge_paper(case_name, context, paper, model=model)["score"]
            )
        except Exception as exc:  # noqa: BLE001 -- an unjudged paper is never scored 0
            judge_failed += 1
            print(f"        ! judge failed for {pid}: {str(exc)[:100]}")
    return {
        **{k: v for k, v in row.items() if k not in ("raw_ids", "raw_titles")},
        # `partial` when something could not be verified or judged: the draw's picks are
        # real, but its TARGET set is a floor rather than a count.
        "status": "partial" if (lookup_failed or judge_failed) else "ok",
        "phase": "judged",
        "targets": sorted(p for p, s in scores.items() if s >= ACTIONABLE),
        "scores": dict(sorted(scores.items())),
        "n_hallucinated": hallucinated,
        "n_lookup_failed": lookup_failed,
        # Not part of the `partial` condition above: an existing-but-abstractless DOI never
        # resolves, so retrying would strand the row forever (C-30).
        "n_unjudgeable": unjudgeable,
        "n_judge_failed": judge_failed,
    }


def load_artifact(prompt_version: str = baseline_mod.DEFAULT_PROMPT_VERSION) -> dict[str, Any]:
    path = out_path(prompt_version)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        # An artifact that does not say which prompt produced it is v1: it predates the
        # versions. Anything that says otherwise is a file being opened under the wrong flag,
        # and merging the two would mix configurations under one set of draw numbers.
        found = stored.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION)
        if found != prompt_version:
            raise SystemExit(
                f"! {path.name} was written under prompt {found!r}, not {prompt_version!r}."
            )
        return stored
    return {
        "_comment": (
            "k independent redraws of the cli baseline over the benchmark25 cohort, judged, "
            "to price how much of the gold set is a property of the draw. Derived by "
            "evals/gold_spread.py; pinned by tests/test_gold_spread.py. Written incrementally "
            "-- a partial artifact is expected and resumable."
        ),
        "cohort": COHORT,
        "draws": DRAWS,
        "prompt_version": prompt_version,
        "results": {},
    }


def save(artifact: dict[str, Any]) -> None:
    path = out_path(artifact.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION))
    path.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8")


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

    version = artifact.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION)
    print(f"prompt: {version}")
    if version != baseline_mod.DEFAULT_PROMPT_VERSION:
        # The pre-registered decision rule at the bottom of this report was registered for
        # redraws of the SAME configuration; it prices sampling noise. A different prompt is
        # a different searcher, so the same arithmetic answers a different question and the
        # rule must not be applied to it. Say so here rather than trusting a reader to notice.
        print(
            "  NOTE: a different searcher, not a redraw. The overlap figures below are\n"
            "  COVERAGE of the frozen set by this configuration, not reproducibility, and\n"
            "  the pre-registered decision rule does NOT apply to them."
        )
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

    # Draws are DISCOVERED, not assumed to be 1..DRAWS: a `--draws 4` trial would otherwise
    # be recorded and silently omitted from every figure below.
    present = sorted({int(k.split("/", 1)[0]) for k in artifact["results"]})
    caps = {}
    for d in present:
        seen = {
            row.get("max_turns", baseline_mod.DEFAULT_MAX_TURNS)
            for key, row in artifact["results"].items()
            if int(key.split("/", 1)[0]) == d
        }
        caps[d] = seen.pop() if len(seen) == 1 else "mixed"
    if any(c != baseline_mod.DEFAULT_MAX_TURNS for c in caps.values()):
        print(f"\nturn cap per draw: {caps}")
        print("  Draws at different caps are DIFFERENT CONFIGURATIONS. The aggregate below")
        print(f"  covers only draws at the shipped cap ({baseline_mod.DEFAULT_MAX_TURNS});")
        print("  others are listed separately so the two are never averaged together.")

    default_draws = [d for d in present if caps[d] == baseline_mod.DEFAULT_MAX_TURNS]
    draws = {d: t for d in default_draws if (t := _draw_targets(artifact, d))}
    other = {d: t for d in present if d not in default_draws and (t := _draw_targets(artifact, d))}
    for d, targets in sorted(other.items()):
        shared = set(targets) & set(frozen)
        hit = sum(len(frozen[c] & targets[c]) for c in shared)
        tot = sum(len(frozen[c]) for c in shared)
        n_t = sum(len(targets[c]) for c in shared)
        print(
            f"  draw {d} @ {caps[d]} turns: {n_t} target(s) over {len(shared)} case(s); "
            f"reproduces {hit}/{tot} of the frozen set"
            if tot
            else f"  draw {d} @ {caps[d]} turns: {n_t} target(s), no frozen overlap"
        )
    if not draws:
        print("\nno complete draws at the shipped cap yet.")
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
        label = "reproducibility" if version == baseline_mod.DEFAULT_PROMPT_VERSION else "coverage"
        print(f"\nmean {label} of the frozen set: {mean:.2f}")
        if version != baseline_mod.DEFAULT_PROMPT_VERSION:
            print("  (different searcher — no verdict; see the note at the top.)")
            return 0
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
    ap.add_argument(
        "--max-turns",
        type=int,
        default=baseline_mod.DEFAULT_MAX_TURNS,
        help="Agent turn cap. Recorded per row; does NOT touch the shipped caches.",
    )
    ap.add_argument(
        "--prompt-version",
        default=baseline_mod.DEFAULT_PROMPT_VERSION,
        choices=sorted(baseline_mod.PROMPTS),
        help=(
            "Which baseline prompt to draw with. v2 allows non-arXiv papers. "
            "Each version writes its OWN artifact; draws are never mixed."
        ),
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent PHASE-A baselines. Phase B stays serial (arXiv + judge are gated).",
    )
    args = ap.parse_args()

    artifact = load_artifact(args.prompt_version)
    if args.report:
        return report(artifact)

    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    cases = cohort_cases(bench)
    if args.case:
        want = set(args.case.split(","))
        if unknown := want - {c["name"] for c in cases}:
            print(f"! unknown case(s): {sorted(unknown)}")
            return 1
        cases = [c for c in cases if c["name"] in want]

    by_key = artifact["results"]
    todo_baseline = [
        (d, c) for d in range(1, args.draws + 1) for c in cases if f"{d}/{c['name']}" not in by_key
    ]
    todo_judge = [k for k, row in by_key.items() if row.get("phase") == "baseline"]

    print(f"cohort: {len(cases)} case(s) x {args.draws} draw(s)")
    print(
        f"recorded: {len(by_key)}   need a baseline: {len(todo_baseline)}   "
        f"need judging: {len(todo_judge)}"
    )
    print(
        f"turn cap: {args.max_turns}   prompt: {args.prompt_version}"
        f"   phase-A concurrency: {args.concurrency}"
    )
    print(f"artifact: {out_path(args.prompt_version).name}")
    if args.dry_run:
        print("\n(dry run)")
        return 0
    if not (todo_baseline or todo_judge):
        print("\nnothing to do; use --report.")
        return report(artifact)

    load_dotenv(EVALS / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        # Picks without verdicts are not targets, and this whole probe is about targets.
        print("! OPENAI_API_KEY is not set; picks would be unjudgeable. Refusing.")
        return 1
    print(f"baseline auth: {baseline_mod.cli_auth_mode()}   judge: {args.model}")

    lock = threading.Lock()

    def record(key: str, row: dict[str, Any]) -> None:
        # After EVERY row, under a lock. This job is too long to lose to one exception, and
        # concurrent writers make a last-writer-wins save a data-loss bug rather than a race
        # nobody notices (C-29 taught the serial version of this lesson).
        with lock:
            artifact["results"][key] = row
            save(artifact)

    # ── phase A: agentic runs, concurrent, no rate-limited service involved ──
    if todo_baseline:
        print(f"\n=== phase A: {len(todo_baseline)} baseline run(s) ===")
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {
                pool.submit(
                    run_baseline_only,
                    case,
                    max_turns=args.max_turns,
                    prompt_version=args.prompt_version,
                ): (draw, case)
                for draw, case in todo_baseline
            }
            for fut in as_completed(futures):
                draw, case = futures[fut]
                key = f"{draw}/{case['name']}"
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 -- one bad case must not end the job
                    print(f"  !! {key} crashed: {type(exc).__name__}: {str(exc)[:140]}")
                    row = {"status": "crashed", "phase": "judged", "raw": str(exc)[:160]}
                record(key, row)
                done += 1
                if row["status"] == "baseline_ok":
                    print(
                        f"  [{done}/{len(todo_baseline)}] {key:<18} {len(row['picks'])} pick(s)"
                        f"  turns={row['num_turns']}  {row['duration_s']}s"
                    )
                else:
                    print(
                        f"  [{done}/{len(todo_baseline)}] {key:<18} !! {row['status']}"
                        f"  {row.get('duration_s', '?')}s"
                    )
        todo_judge = [k for k, row in artifact["results"].items() if row.get("phase") == "baseline"]

    # ── phase B: verify + judge, serial, because arXiv and the judge are gated ──
    if todo_judge:
        print(f"\n=== phase B: judging {len(todo_judge)} run(s) (serial) ===")
        for n, key in enumerate(sorted(todo_judge), start=1):
            case_name = key.split("/", 1)[1]
            try:
                row = judge_row(case_name, artifact["results"][key], model=args.model)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! {key} crashed while judging: {type(exc).__name__}: {str(exc)[:140]}")
                continue
            record(key, row)
            print(
                f"  [{n}/{len(todo_judge)}] {key:<18} {len(row['targets'])} target(s)"
                f"  [{row['status']}]"
            )

    print(f"\nwrote {OUT.name}\n")
    return report(artifact)


if __name__ == "__main__":
    raise SystemExit(main())
