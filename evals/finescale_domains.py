"""Is the fine-scale map calibrated on scientific software? ($0, offline, from artifacts.)

    uv run python evals/finescale_domains.py
    uv run python evals/finescale_domains.py --sci <run.json> --legacy <run.json>

§8's plan item 3 named `calibrate_finescale.py --analyse` for this and said it was "$0 extra".
**The cost was right and the instrument was wrong.** `--analyse` re-reads that script's own
per-case cache under `.work/calibration/`, which holds the 22 legacy cases from the 2026-08-09
run and none of the twelve scientific ones; pointed at the cohort-3 artifact it prints
`0/15 cached` twelve times and analyses nothing. Re-scoring into that cache means cloning
twelve repositories and paying for a fresh gate and fine-scale pass.

None of that is necessary. `run_judge_eval._apply_finescale` mutates the ranked window in
place, so **the artifact already carries `finescale` and `finescale_p` for every score-2 band
paper** — 324 of 555 papers across the 37-case run, with no gaps and none outside the band.
This script reads them and hands them to `calibrate_finescale.analyse`, which is the same
analysis the paid path runs. No API keys, no clones, no cache.

**The reproduction check comes first**, for the reason the paid script gives: if the rebuilt
policy does not arrive at the Top Picks the live run recorded, the analysis is measuring itself.

**What this can and cannot settle.** The judge-free half — how often the map withholds, where
it puts its probabilities — is a fact about the map. The judge-dependent half is scored against
one judge's labels, and `evals/second_judge.py` measured that judge against Sonnet at kappa
0.507. So this script prints the two halves under separate headings and projects every
judge-dependent conclusion through the second-judge transition rates before stating it. §17.2
is why: a claim about the fine-scale stage that rests on GPT-scored 2s is resting on the one
cell where the two judges agree 8 times in 48.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))

from calibrate_finescale import (  # noqa: E402
    ACTIONABLE,
    INTERCEPT,
    MIN_ACTIONABLE,
    SLOPE,
    analyse,
    base_id,
    net2,
    shown_by_policy,
)
from run_judge_eval import RESULTS_DIR  # noqa: E402
from why_case import SECOND_JUDGE  # noqa: E402

# The cohort-3 session: twelve scientific cases and the legacy re-baseline, judged hours
# apart against frozen pools. Paired by construction — same pipeline, same judge, same day.
DEFAULT_SCI = "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T060917Z.json"
DEFAULT_LEGACY = "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T172033Z.json"

# P(second judge scores >= 2 | this judge's score), read off §17.2's transition table.
# SECOND_JUDGE states rows 0 and 1 as "stays at or below", rows 2 and 3 as ">= 2", so the
# complement is taken for the first two rather than the strings reused blindly.
P_SONNET_ACTIONABLE = {0: 0.0, 1: 3 / 61, 2: 8 / 48, 3: 32 / 33}

BREAK_EVEN = 2 / 3  # net@2 prices a shown paper at 3p - 2, so showing pays above exactly this


def load_population(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    """Ranked window and recorded Top Picks per case, straight from a results artifact."""
    data: dict[str, list[dict[str, Any]]] = {}
    recorded: dict[str, set[str]] = {}
    for rec in json.loads(path.read_text(encoding="utf-8")):
        case = rec["case"]
        data[case] = rec["returned"]["reporadar_top10"]
        recorded[case] = {base_id(p["arxiv_id"]) for p in rec["returned"]["reporadar_toppicks"]}
    return data, recorded


def band_of(data: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        r
        for case in data
        for r in data[case]
        if r.get("llm_score") == MIN_ACTIONABLE and r.get("finescale_p") is not None
    ]


def stage_value(data: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
    """net@2 per case with the fine-scale stage and without it.

    "Without it" is not a hypothetical: it is `sweep_top_picks` at min>=2, which re-gates the
    *same* ranked window on the *same* triage scores. The two selections differ by exactly one
    shipped stage, which is what makes the difference readable as that stage's contribution.
    Verified equal to the recorded `reporadar_toppicks_sweep["2"]` net value on all 37 cases.
    """
    cases = sorted(data)
    with_stage = sum(
        net2([r for r in data[c] if shown_by_policy(r, SLOPE, INTERCEPT)]) for c in cases
    )
    without = sum(
        net2([r for r in data[c] if (r.get("llm_score") or -1) >= MIN_ACTIONABLE]) for c in cases
    )
    return with_stage / len(cases), without / len(cases)


def check_sweep_equivalence(path: Path) -> tuple[int, int, list[str]]:
    """Is "the stage removed" really the recorded sweep arm, case by case?

    §15.5 set the `--rr-sweep min>=2` observation aside as returning "more papers from a
    20-candidate rerank pool". It does not: `sweep_top_picks` filters the same ranked window
    on `llm_score`, so min>=2 is the shipped Top Picks *before* `_apply_finescale` runs. That
    reading is what lets this script call the difference one stage's contribution, so it is
    checked against the recorded numbers rather than argued from the source.
    """
    agree = total = 0
    bad: list[str] = []
    for rec in json.loads(path.read_text(encoding="utf-8")):
        recorded = rec.get("reporadar_toppicks_sweep", {}).get("2", {}).get("net_value@2")
        if recorded is None:
            continue
        total += 1
        rebuilt = net2(
            [
                r
                for r in rec["returned"]["reporadar_top10"]
                if (r.get("llm_score") or -1) >= MIN_ACTIONABLE
            ]
        )
        if abs(rebuilt - recorded) < 1e-9:
            agree += 1
        else:
            bad.append(f"{rec['case']}: rebuilt {rebuilt:+.1f} vs recorded {recorded:+.1f}")
    return agree, total, bad


def judge_free(label: str, data: dict[str, list[dict[str, Any]]]) -> None:
    band = band_of(data)
    held = [r for r in band if not shown_by_policy(r, SLOPE, INTERCEPT)]
    ps = sorted(r["finescale_p"] for r in band)
    print(
        f"  {label:14} band={len(band):3d}  withheld={len(held):3d} ({len(held) / len(band):4.0%})"
        f"  mean_p={sum(ps) / len(ps):.3f}  median_p={ps[len(ps) // 2]:.3f}"
        f"  mean_expectation={sum(r['finescale'] for r in band) / len(band):.2f}"
    )


def judge_swap(label: str, data: dict[str, list[dict[str, Any]]]) -> None:
    """Project the withheld set through the second-judge transition rates.

    A projection, not a measurement: it applies marginal rates from a 200-label stratified
    sample to a different 33-47 papers. It cannot replace second-judging these papers. It is
    here because it is the cheap check that would have caught §17.2 before it was written.
    """
    held = [r for r in band_of(data) if not shown_by_policy(r, SLOPE, INTERCEPT)]
    hist: dict[int, int] = {}
    for r in held:
        hist[r["judge_score"]] = hist.get(r["judge_score"], 0) + 1
    observed = sum(n for j, n in hist.items() if j >= ACTIONABLE) / len(held)
    projected = sum(P_SONNET_ACTIONABLE[j] * n for j, n in hist.items()) / len(held)
    print(f"  {label:14} withheld n={len(held):3d}  judge {dict(sorted(hist.items()))}")
    print(
        f"  {'':14}   actionable as judged: {observed:5.0%}   projected under a second judge:"
        f" {projected:5.0%}   (break-even {BREAK_EVEN:.0%})"
    )
    for j in sorted(hist):
        frac, pct, note = SECOND_JUDGE[j]
        print(f"  {'':14}     judge-{j}: {hist[j]:2d} paper(s) — {frac} ({pct}) {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sci", default=DEFAULT_SCI)
    ap.add_argument("--legacy", default=DEFAULT_LEGACY)
    args = ap.parse_args()

    pops = {}
    paths = {}
    for label, name in (("SCIENTIFIC-12", args.sci), ("LEGACY-25", args.legacy)):
        path = Path(name) if Path(name).is_file() else RESULTS_DIR / name
        paths[label] = path
        pops[label] = load_population(path)
        print(f"{label:14} {path.name}  ({len(pops[label][0])} cases)")

    print("\n" + "=" * 78)
    print("REPRODUCTION — nothing below is readable until this passes")
    print("=" * 78)
    ok = True
    for label, (data, recorded) in pops.items():
        out = analyse(data, recorded)
        rep = out["reproduction"]
        bad = [c for c in rep["cases"] if c["only_rebuilt"] or c["only_live"]]
        print(
            f"  {label:14} rebuilt Top Picks agree with the live run: {rep['agree']}/{rep['live']}"
        )
        for c in bad:
            ok = False
            print(f"    ! {c['case']}: only_rebuilt={c['only_rebuilt']} only_live={c['only_live']}")
        pops[label] = (data, recorded, out)

    print("\n  'stage removed' vs the recorded --rr-sweep min>=2 arm:")
    for label, path in paths.items():
        agree, total, bad = check_sweep_equivalence(path)
        print(f"    {label:14} {agree}/{total} cases identical")
        for line in bad:
            ok = False
            print(f"      ! {line}")
    if not ok:
        print("\n  REPRODUCTION FAILED — the rebuild does not reach the live decision.")
        return 1

    print("\n" + "=" * 78)
    print("JUDGE-FREE — what the map does, independent of any label")
    print("=" * 78)
    for label, (data, _, _) in pops.items():
        judge_free(label, data)

    print("\n" + "=" * 78)
    print("JUDGE-DEPENDENT — scored against GPT-5.5 labels (kappa 0.507 vs Sonnet)")
    print("=" * 78)
    for label, (data, _, out) in pops.items():
        cal = out["calibration"]["band"]
        with_stage, without = stage_value(data)
        print(f"\n  {label}")
        print(
            f"    base_rate={cal['base_rate']:.3f}  mean_p={cal['mean_p']:.3f}"
            f"  ECE={cal['ece']:.3f}  AUC={cal['auc']:.3f}  Brier={cal['brier']:.3f}"
        )
        for t in cal["reliability"]:
            print(
                f"      {t['bin']}  n={t['n']:3d}  mean_p={t['mean_p']:.3f}"
                f"  empirical={t['empirical']:.3f}"
            )
        print(f"    net@2/case with the fine-scale stage : {with_stage:+.3f}")
        print(
            f"    net@2/case with the stage removed    : {without:+.3f}"
            f"   (the stage is worth {with_stage - without:+.3f})"
        )
        rf = out["refit"]
        st = rf["sign_test"]
        print(
            f"    LORO refit delta {rf['mean_delta']:+.3f}/case  ci95="
            f"[{rf['ci95'][0]:+.3f}, {rf['ci95'][1]:+.3f}]  "
            f"sign {st['pos']}+/{st['neg']}-/{st['ties']}= p={st['p']:.3f}"
        )

    print("\n" + "=" * 78)
    print("JUDGE-SWAP CHECK — does the judge-dependent half survive a second judge?")
    print("=" * 78)
    for label, (data, _, _) in pops.items():
        judge_swap(label, data)
    print(
        "\n  Read this before quoting anything above it. net@2 is DEFINED on judge labels, so\n"
        "  'the stage costs the benchmark N points' is true by construction. 'The stage\n"
        "  withholds actionable papers' is a different claim and needs the projection to clear\n"
        f"  {BREAK_EVEN:.0%}. Where it does not, the honest statement is that the gate and the\n"
        "  map disagree with one judge in the region where two judges disagree with each other."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
