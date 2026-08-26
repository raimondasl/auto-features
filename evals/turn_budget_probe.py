"""Is the baseline's 12-turn cap an active constraint, or slack headroom? [P15]

Four `bio-*`/`mat-*` cases return `error_max_turns` against the `--max-turns 12` in
`CLAUDE_FLAGS` (P14) -- a third of that cohort, against 2 of the original 25. The obvious fix
is to raise the cap, and it is not free: `_discriminator` hashes the flags, so raising it
re-runs **all 37** cases and re-derives the gold set every published recall denominator
divides by. That is the 2026-08-09 incident, which moved `graph` from 3 targets to 4.

So the question to answer first is not "is 30 better" but **"does 30 change anything on the
cases that already succeed at 12"** -- because if it does not, the re-run is a restatement,
and if it does, it is a rebuild.

**The confound, and the arm that handles it.** The baseline is not deterministic: the same
case re-run under the same flags does not necessarily return the same papers. So comparing a
30-turn run against the *stored* 12-turn answer cannot distinguish "the cap mattered" from
"it is a different draw". Every case therefore gets a **fresh 12-turn control** alongside the
30-turn arm, run back to back under the same auth, and the treatment is judged against that
control rather than against the cache. The cache-vs-control distance is what re-running alone
costs, and it is the yardstick the treatment has to beat.

    cached  (A)  the stored answer -- older session, and for `bio-*`/`mat-*` a different auth
    control (B)  fresh, 12 turns, current auth   <- measures nondeterminism
    treat   (C)  fresh, 30 turns, current auth   <- the only variable against B

**Pre-registered, written before the first call:**

* **Primary measure: whether the run hit the cap, read off `status`.** A run that reaches
  `--max-turns` does not truncate quietly -- it fails with `subtype: error_max_turns`, which
  is exactly how P14's four cases present. So a control arm that returns `ok` at 12 turns
  did not reach the cap, and one that fails did.
* **Secondary measure.** Jaccard of the recommended id sets, version-stripped, compared
  PAIRED per case: J(B,C) against J(A,B). The question is not whether the turn change moves
  picks -- everything moves picks -- but whether it moves them more than a re-run does.
* **Decision rule.** *Restatement* iff no control arm fails at 12 AND the paired difference
  J(B,C) - J(A,B) is not resolvably negative. *Rebuild* iff the turn change moves picks
  resolvably more than a re-run. **Inconclusive** otherwise, which at n = 6 against this
  benchmark's known noise is the outcome to expect and must be reportable rather than
  rounded to one of the other two.
* **Kill condition.** If the control arm itself fails on a case the cache says succeeded,
  the 12-turn boundary is being reached nondeterministically; report it and treat every
  similarity number in the run as uninterpretable.

**A retracted measure, recorded rather than deleted [C-27].** The first version of this
pre-registration made `num_turns` the primary measure, on the reasoning that a 30-turn arm
which never exceeds 12 turns proves the cap was slack. **`num_turns` is not the quantity
`--max-turns` bounds**, and the run disproved it immediately: every *control* arm, capped at
12, reported `num_turns` of 16, 17 or 9 and still returned `ok`. Measured directly,
`--max-turns 2` on a tool-using prompt fails with `error_max_turns` at `num_turns: 3` and
`--max-turns 12` on the real baseline prompt SUCCEEDS at `num_turns: 15`, so the cap is
enforced and the two numbers count different things. What each counts is not defined by the
payload; the operational rule is to read `status`, never `num_turns`.
The original rule would have read "6 of 6 cases exceeded the cap" and
declared a rebuild. It is retained above as an observation with that caveat attached, never
as a threshold. Note the direction: the correction makes the answer *less* decisive, not more
convenient.

**Case selection, fixed before running.** Three per cohort, spanning the outcome types, so
no result can be attributed to a convenient subset:

| case | repo | cohort | why |
|---|---|---|---|
| `rag` | ColBERT | bench25 | most productive; if the cap binds anywhere, here |
| `linter` | ruff | bench25 | the net-negative case: does budget buy *quality*? |
| `http` | requests | bench25 | does budget turn "nothing" into something? |
| `mat-descriptors` | dscribe | scisoft | the productive scientific case |
| `bio-align` | minimap2 | scisoft | a typical scientific case |
| `bio-singlecell` | scanpy | scisoft | declined explicitly, with reasons |

**These are the cases that already SUCCEED at 12 turns.** The four that fail
(`bio-scvi`/scvi-tools, `mat-mlip`/MACE, `mat-toolkit`/pymatgen, `mat-phonon`/phonopy) are a
separate question -- does raising the cap *rescue* them -- run with `--case` and `--out`.

The cohort split is a second, free experiment: the benchmark25 caches were written under a
signed-in CLI and the scientific ones under ANTHROPIC_API_KEY (P14, and the open question
left by the auth change). If A-vs-B similarity is comparable across the two cohorts, the auth
path does not change what the agent recommends.

**Nothing is written to the shared caches.** Every arm runs with `use_cache=False`, so the
33 stored answers are neither read nor overwritten -- `_cache_path` has no discriminator in
it, so a 30-turn run would otherwise *replace* the 12-turn answer it is being compared to.

    uv run python evals/turn_budget_probe.py --dry-run   # $0, the plan and the arms
    uv run python evals/turn_budget_probe.py             # 12 agentic runs, subscription
    uv run python evals/turn_budget_probe.py --report    # $0, re-read the artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402
from harness import WORK_DIR, assemble_repo_context, clone_repo  # noqa: E402
from run_judge_eval import load_dotenv  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
CLI_CACHE = EVALS / "cache" / "baseline" / "cli"
OUT = EVALS / "turn_budget_probe.json"

CONTROL_TURNS, TREATMENT_TURNS = 12, 30

# Fixed before the first call; see the table in the module docstring for the rationale.
CASES = ["rag", "linter", "http", "mat-descriptors", "bio-align", "bio-singlecell"]


def _flags(max_turns: int) -> list[str]:
    """`CLAUDE_FLAGS` with the turn cap replaced -- built from the shipped list, not retyped."""
    flags = list(baseline_mod.CLAUDE_FLAGS)
    i = flags.index("--max-turns")
    flags[i + 1] = str(max_turns)
    return flags


def _ids(result: dict[str, Any]) -> set[str]:
    return {dedup_id(i) for i in result.get("ids") or []}


def _jaccard(a: set[str], b: set[str]) -> float | None:
    """None when both sides are empty -- two abstentions agree, but not on any *content*."""
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def cached_arm(case: str) -> dict[str, Any]:
    path = CLI_CACHE / f"{case}.json"
    if not path.is_file():
        return {"status": "missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("raw") or ""
    # Replay rather than trust the stored ids, so this arm is what the harness would score
    # today (the C-25 rule), including the block-keyed fallback for the damaged caches.
    ids, _ = baseline_mod._parse_recommendations(raw)
    if not ids and not baseline_mod._has_answer_block(raw):
        ids = list(data.get("ids") or [])
    return {
        "status": data.get("status"),
        "ids": ids,
        "num_turns": data.get("num_turns"),
        "cli_auth_mode": data.get("cli_auth_mode"),
    }


def run_arm(case: str, repo: Path, context: str, max_turns: int) -> dict[str, Any]:
    out = baseline_mod.run_baseline(
        repo,
        repo_name=case,
        repo_context=context,
        mode="cli",
        use_cache=False,  # never read, never write -- the stored answers are the comparison
        flags=_flags(max_turns),
    )
    return {
        "status": out.get("status"),
        "ids": out.get("ids") or [],
        "num_turns": out.get("num_turns"),
        "cost_usd": out.get("cost_usd", 0.0),
        "max_turns": max_turns,
        "raw_head": (out.get("raw") or "")[:180],
    }


def build(cases: list[str]) -> dict[str, Any]:
    rows = []
    for n, case in enumerate(cases, start=1):
        print(f"\n[{n}/{len(cases)}] {case}")
        repo = WORK_DIR / case
        if not repo.is_dir():
            print("      ! no clone; skipping")
            rows.append({"case": case, "status": "no_clone"})
            continue
        context = assemble_repo_context(repo)
        cached = cached_arm(case)
        control = run_arm(case, repo, context, CONTROL_TURNS)
        print(
            f"      control  ({CONTROL_TURNS}t): {control['status']:<14} "
            f"turns={control['num_turns']} picks={len(control['ids'])}"
        )
        treat = run_arm(case, repo, context, TREATMENT_TURNS)
        print(
            f"      treat    ({TREATMENT_TURNS}t): {treat['status']:<14} "
            f"turns={treat['num_turns']} picks={len(treat['ids'])}"
        )
        rows.append(
            {
                "case": case,
                "cohort": "scisoft" if case.startswith(("bio-", "mat-")) else "benchmark25",
                "cached": cached,
                "control": control,
                "treat": treat,
                "j_cached_control": _jaccard(set(cached.get("ids") or []), _ids(control)),
                "j_control_treat": _jaccard(_ids(control), _ids(treat)),
            }
        )
    return {
        "_comment": (
            "P15: does raising --max-turns 12 -> 30 change what the baseline recommends on "
            "cases that already succeed at 12? Arms run with use_cache=False; the shared "
            "caches are untouched. See evals/turn_budget_probe.py for the pre-registration."
        ),
        "control_turns": CONTROL_TURNS,
        "treatment_turns": TREATMENT_TURNS,
        "cases": rows,
    }


def report(data: dict[str, Any]) -> int:
    rows = [r for r in data["cases"] if "control" in r]
    if not rows:
        print("no usable rows.")
        return 1

    header = (
        f"\n{'case':<17}{'coh':<12}{'cached':>8}{'ctrl':>16}{'treat':>16}{'J(A,B)':>9}{'J(B,C)':>9}"
    )
    print(header)
    for r in rows:
        c, t = r["control"], r["treat"]
        jab = "  n/a" if r["j_cached_control"] is None else f"{r['j_cached_control']:.2f}"
        jbc = "  n/a" if r["j_control_treat"] is None else f"{r['j_control_treat']:.2f}"
        ctrl_cell = f"{len(c['ids'])}p/{c['num_turns']}t"
        treat_cell = f"{len(t['ids'])}p/{t['num_turns']}t"
        n_cached = len(r["cached"].get("ids") or [])
        print(
            f"{r['case']:<17}{r['cohort']:<12}{n_cached:>8}"
            f"{ctrl_cell:>16}{treat_cell:>16}{jab:>9}{jbc:>9}"
        )

    # --- the kill condition, checked before anything is interpreted -----------------
    broke = [r["case"] for r in rows if r["control"]["status"] != "ok"]
    if broke:
        print(f"\n!! KILL CONDITION: the 12-turn control failed on {broke}.")
        print("   The cap is being reached nondeterministically; treat every J below as")
        print("   uninterpretable and do not read a verdict off this run.")
        return 1

    # --- primary measure: did any arm actually REACH the cap? ----------------------
    # Read off `status`, not `num_turns`. Reaching `--max-turns` fails loudly with
    # `error_max_turns` (that is how P14's four cases present), so `ok` at 12 means the cap
    # was never reached. `num_turns` is NOT the quantity the cap bounds -- controls capped
    # at 12 report 16 and 17 while succeeding -- so it is printed and never thresholded.
    capped = [r["case"] for r in rows if r["control"]["status"] != "ok"]
    print(f"\nreached the {CONTROL_TURNS}-turn cap (control arm): {capped or 'none'}")
    if not capped:
        print(f"  -> the cap bound on no case probed; all {len(rows)} controls finished.")
    obs = ", ".join(
        f"{r['case']}={r['control']['num_turns']}/{r['treat']['num_turns']}" for r in rows
    )
    print(f"observed num_turns (control/treat): {obs}")
    print(f"  NB [C-27]: not comparable to --max-turns={CONTROL_TURNS}; different quantities.")

    # --- the secondary measure, PAIRED, against the yardstick it must beat ----------
    paired = [
        (r["case"], r["j_control_treat"] - r["j_cached_control"])
        for r in rows
        if r["j_control_treat"] is not None and r["j_cached_control"] is not None
    ]
    jab = [r["j_cached_control"] for r in rows if r["j_cached_control"] is not None]
    jbc = [r["j_control_treat"] for r in rows if r["j_control_treat"] is not None]
    m_ab = sum(jab) / len(jab) if jab else float("nan")
    m_bc = sum(jbc) / len(jbc) if jbc else float("nan")
    print(f"\nmean J(cached, control) = {m_ab:.2f}   <- what re-running ALONE costs")
    print(f"mean J(control, treat)  = {m_bc:.2f}   <- what the turn change costs on top")
    print(
        f"  -> a re-run of the identical configuration already disagrees with the stored\n"
        f"     answer on ~{(1 - m_ab) * 100:.0f}% of picks. That is the noise any turn effect\n"
        f"     has to clear."
    )

    deltas = [d for _, d in paired]
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        lo, hi = paired_bootstrap(deltas)
        print(f"\npaired J(B,C) - J(A,B) over {len(deltas)} case(s): {mean_d:+.2f}")
        print("  per case: " + ", ".join(f"{c} {d:+.2f}" for c, d in paired))
        print(f"  bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
        resolvable = hi < 0 or lo > 0
    else:
        mean_d, resolvable = 0.0, False

    for cohort in ("benchmark25", "scisoft"):
        vals = [
            r["j_cached_control"]
            for r in rows
            if r["cohort"] == cohort and r["j_cached_control"] is not None
        ]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  J(cached, control), {cohort:<12}: {mean:.2f}  (n={len(vals)})")
    print("  (the two cohorts' caches were written under different CLI auth; comparable")
    print("   values here mean the auth path does not change what the agent recommends)")

    if capped:
        call = "REBUILD -- the cap binds even on cases the cache says succeed"
    elif not resolvable:
        call = (
            f"INCONCLUSIVE -- the turn effect ({mean_d:+.2f}) is inside the noise of simply\n"
            f"         re-running, at n={len(deltas)}. The probe cannot separate them; a\n"
            f"         bigger effect or many more paired draws would be needed"
        )
    elif mean_d >= 0:
        call = "RESTATEMENT -- picks move no more than a re-run moves them"
    else:
        call = "REBUILD -- the turn change moves picks resolvably more than a re-run"
    print(f"\nVERDICT: {call}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe whether the 12-turn baseline cap binds.")
    ap.add_argument("--case", help="Comma-separated override of the pre-registered set.")
    ap.add_argument("--dry-run", action="store_true", help="$0: print the plan.")
    ap.add_argument("--report", action="store_true", help="$0: re-read the stored artifact.")
    ap.add_argument("--out", help="Write/read a different artifact than the default.")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else OUT

    if args.report:
        if not out_path.is_file():
            print(f"no artifact at {out_path}")
            return 1
        return report(json.loads(out_path.read_text(encoding="utf-8")))

    cases = args.case.split(",") if args.case else list(CASES)
    load_dotenv(EVALS / ".env")
    auth = baseline_mod.cli_auth_mode()
    print(f"cases ({len(cases)}): {', '.join(cases)}")
    print(f"arms per case: control {CONTROL_TURNS} turns, treatment {TREATMENT_TURNS} turns")
    print(f"auth: {auth}  |  caches: NOT read, NOT written (use_cache=False)")
    if args.dry_run:
        for case in cases:
            a = cached_arm(case)
            print(f"  {case:<17} cached: {a.get('status')} {len(a.get('ids') or [])} pick(s)")
        print("\n(dry run)")
        return 0

    for case in cases:
        if not clone_repo("", WORK_DIR / case, reuse=True) and not (WORK_DIR / case).is_dir():
            print(f"! no clone for {case}; run the benchmark once first.")
            return 1

    data = build(cases)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    return report(data)


if __name__ == "__main__":
    raise SystemExit(main())
