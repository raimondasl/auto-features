"""What is the smallest effect this benchmark can actually resolve?

    uv run python evals/noise_floor.py A.json B.json      # $0, reads two completed runs

Several conclusions in this project are *null* results — "not significant", "the interval
spans zero" — and a null is only informative if the instrument could have detected the
effect had it been there. That was never checked. This script checks it, from two runs of
the **same configuration**, by treating their per-case difference as pure noise: identical
inputs, identical flags, so everything that moves is the benchmark's own variance.

Measured over **three draws** of the shipped config, 22 shared cases, 2026-08-08/10/11
(42 df): residual per-case sd **1.23**, whole-run shift sd **0.27**, and a **minimum
resolvable effect of 1.04 net@2 per case** for a paired same-session comparison (1.07
against a stored run, where the whole-run shift no longer cancels). The third draw barely
moved the estimate the first two gave — 1.03 → 1.04 — so the floor is now a number with
some weight behind it rather than one difference.

The dominant term is *which papers were collected*: the ranked top-10 overlaps only
**0.49** by Jaccard between draws, so roughly a third of what a run shows is different next
time. Eight of 22 cases never move at all; the three noisiest carry 47% of the variance.

**The consequence is a sizing rule.** An experiment targeting a mean effect below ~1.0
net@2 cannot be resolved here at n≈25, however carefully it is run. Two of this project's
own experiments were below that floor before they started (stated-intent goals at +0.44,
the register-flip arm at +0.12); their nulls are real but uninformative, and the right
response was a more sensitive instrument, not a firmer conclusion. Freezing the candidate
pool (`--rr-frozen-pool`) removes the dominant term for any treatment *downstream* of
retrieval.

**A single run's p-value is not a property of the system.** The same 25-case configuration
scored p = 0.0414 (15 w / 5 l) on one draw and p = 0.0001 (18 w / 1 l) two days later. Report
a mean over draws, or an interval — never one run's significance as if it were stable.

Cross-run comparisons are only meaningful when both runs collected live, or when both
reused the *same* frozen pool. This script refuses to mix them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_judge_eval import RESULTS_DIR  # noqa: E402


def load(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        p = RESULTS_DIR / path
    return {r["case"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def provenance(run: dict[str, dict[str, Any]]) -> str:
    """'live', 'frozen:<fingerprint>', or 'mixed'/'unlabelled' — see --rr-frozen-pool."""
    seen = {
        (r.get("pool_provenance") or {}).get("mode", "unlabelled")
        + (
            ":" + (r.get("pool_provenance") or {}).get("fingerprint", "")[:12]
            if (r.get("pool_provenance") or {}).get("mode") == "frozen"
            else ""
        )
        for r in run.values()
    }
    return seen.pop() if len(seen) == 1 else "mixed"


def jaccard(a: set[str], b: set[str]) -> float | None:
    return len(a & b) / len(a | b) if (a or b) else None


def ids(rec: dict[str, Any], key: str) -> set[str]:
    return {p["arxiv_id"].split("v")[0] for p in rec["returned"].get(key, [])}


def decompose(by_case: dict[str, list[float]]) -> dict[str, float]:
    """Split run-to-run variation into a whole-run shift and per-case residual noise.

    The distinction decides which number sizes an experiment, and the two differ here.
    A benchmark mean moves for two reasons:

    * a **draw effect** — the whole run shifts together (arXiv's index changed, the day's
      collection was richer). Between the first two draws this was **−0.64 net@2 across
      every case at once**.
    * **residual** per-case noise — this repo happened to surface a different paper.

    In a *paired same-session* comparison the draw effect is common to both arms and
    largely cancels, so residual noise sets the resolution. In a comparison against a
    **stored** run from another day it does not cancel and both terms count. That is the
    quantitative case for the paired-arm discipline every experiment here follows, and it
    is why a stored-run control was rejected for the HyDE evaluation.

    Two-way layout without replication: x_ij = mu + case_i + draw_j + e_ij.
    """
    cases = sorted(by_case)
    k = min(len(v) for v in by_case.values())
    x = {c: by_case[c][:k] for c in cases}
    n = len(cases)
    grand = statistics.mean(v for c in cases for v in x[c])
    case_eff = {c: statistics.mean(x[c]) - grand for c in cases}
    draw_eff = [statistics.mean(x[c][j] for c in cases) - grand for j in range(k)]
    ss_resid = sum(
        (x[c][j] - grand - case_eff[c] - draw_eff[j]) ** 2 for c in cases for j in range(k)
    )
    df_resid = (n - 1) * (k - 1)
    sd_resid = math.sqrt(ss_resid / df_resid) if df_resid else 0.0
    sd_draw = statistics.pstdev(draw_eff) if k > 1 else 0.0
    return {
        "k": k,
        "n": n,
        "sd_resid": sd_resid,
        "df_resid": df_resid,
        "sd_draw": sd_draw,
        "draw_range": (max(draw_eff) - min(draw_eff)) if k > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "runs", nargs="+", metavar="RUN", help="two or more results files of the SAME config"
    )
    ap.add_argument(
        "--assume-unlabelled-live",
        action="store_true",
        help="Treat runs with no `pool_provenance` as live collections. Runs written before "
        "--rr-frozen-pool existed could not have been frozen, so this is sound for them — but "
        "it is an assumption about data the artifact does not contain, so it must be stated on "
        "the command line rather than inferred silently, and it is NOT backfilled into the "
        "files themselves.",
    )
    args = ap.parse_args()
    if len(args.runs) < 2:
        raise SystemExit("need at least two runs of the same configuration")

    runs = [load(x) for x in args.runs]
    provs = [provenance(r) for r in runs]
    for name, prov in zip(args.runs, provs, strict=True):
        print(f"  {Path(name).name}: {prov}")
    if args.assume_unlabelled_live and set(provs) <= {"live", "unlabelled"}:
        print(
            "  !! --assume-unlabelled-live: treating unlabelled runs as live collections "
            "(asserted by the caller, not recorded in the artifacts)"
        )
        provs = ["live"] * len(provs)
    if len(set(provs)) > 1:
        raise SystemExit(
            f"refusing to compare runs with different pool provenance ({sorted(set(provs))}): "
            "their candidate pools were produced differently, so the difference between "
            "them is not this benchmark's noise."
        )

    shared = sorted(set.intersection(*(set(r) for r in runs)))
    if len(shared) < 3:
        raise SystemExit(f"only {len(shared)} shared cases — not enough to estimate anything")

    # --- The direct estimate, available from any number of draws -----------------------
    by_case = {
        c: [r[c]["reporadar_toppicks"]["net_value@2"] for r in runs if c in r] for c in shared
    }
    d = decompose(by_case)
    n = d["n"]
    # Same session: the draw effect is common to both arms and cancels. Against a stored
    # run from another day it does not, and both terms count.
    se_paired = d["sd_resid"] * math.sqrt(2) / math.sqrt(n)
    se_stored = math.sqrt(2 * (d["sd_resid"] ** 2 + d["sd_draw"] ** 2)) / math.sqrt(n)
    print(f"\n{int(d['k'])} draws x {n} shared cases, identical configuration")
    print(f"  residual sd (per case, per draw) : {d['sd_resid']:.2f}   ({int(d['df_resid'])} df)")
    print(
        f"  whole-run shift between draws    : sd {d['sd_draw']:.2f}, "
        f"spread {d['draw_range']:.2f} net@2"
    )
    print("\n  MINIMUM RESOLVABLE EFFECT (80% power, two-sided 0.05):")
    print(f"    paired, same session (draw effect cancels) : {2.8 * se_paired:.2f} net@2/case")
    print(f"    against a STORED run (it does not)         : {2.8 * se_stored:.2f} net@2/case")

    print(f"\n  per-case mean and spread over {len(runs)} draws (noisiest first):")
    rows = sorted(
        ((statistics.pstdev(v), statistics.mean(v), c) for c, v in by_case.items()), reverse=True
    )
    for s, m, c in rows[:8]:
        vals = " ".join(f"{v:+.0f}" for v in by_case[c])
        print(f"    {c:11} mean {m:+5.1f}  sd {s:4.2f}   draws: {vals}")
    stable = sum(1 for s, _, _ in rows if s == 0)
    print(f"    ... {stable}/{n} cases identical across every draw")
    total = sum(s * s for s, _, _ in rows)
    if total:
        top = sum(s * s for s, _, _ in rows[:3])
        print(f"    top 3 noisiest cases carry {top / total:.0%} of the total variance")

    if len(runs) == 2:
        a, b = runs
        d = [
            b[c]["reporadar_toppicks"]["net_value@2"] - a[c]["reporadar_toppicks"]["net_value@2"]
            for c in shared
        ]
        print(
            f"\n  (two-draw view: per-case delta mean {statistics.mean(d):+.2f}, "
            f"sd {statistics.stdev(d):.2f})"
        )

    # --- Where the noise comes from: how much of a run's output is even the same set? ---
    print("\n  candidate churn, averaged over every pair of draws:")
    for key, label in (("reporadar_toppicks", "shown (Top Picks)"), ("reporadar_top10", "top-10")):
        vals = [
            j
            for i, x in enumerate(runs)
            for y in runs[i + 1 :]
            for c in shared
            if (j := jaccard(ids(x[c], key), ids(y[c], key))) is not None
        ]
        if vals:
            print(f"    Jaccard, {label:18}: {statistics.mean(vals):.3f}")

    print("\n  Sizing rule: an experiment aiming below the minimum resolvable effect cannot")
    print("  be answered by this instrument at this n. Freeze the pool (--rr-frozen-pool)")
    print("  when the treatment is downstream of retrieval, or accept an uninformative null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
