"""What is the smallest effect this benchmark can actually resolve?

    uv run python evals/noise_floor.py A.json B.json      # $0, reads two completed runs

Several conclusions in this project are *null* results — "not significant", "the interval
spans zero" — and a null is only informative if the instrument could have detected the
effect had it been there. That was never checked. This script checks it, from two runs of
the **same configuration**, by treating their per-case difference as pure noise: identical
inputs, identical flags, so everything that moves is the benchmark's own variance.

Measured 2026-08-10 on the shipped config, 22 shared cases two days apart: per-case
sd **1.73**, standard error of the 22-case mean **0.37**, so a 95% interval on a single
benchmark mean is about **±0.73**. Half of that comes from *which papers were collected* —
the ranked top-10 overlaps only **0.50** by Jaccard between the two runs, so roughly a
third of the papers a run shows are different next time.

**The consequence is a sizing rule.** An experiment targeting a mean effect below roughly
0.7 net@2 cannot be resolved here at n≈25, no matter how carefully it is run. Two of this
project's own experiments were below that floor before they started (stated-intent goals at
+0.44, the register-flip arm at +0.12); their nulls are real but uninformative, and the
right response was a more sensitive instrument, not a bigger conclusion. Freezing the
candidate pool (`--rr-frozen-pool`) removes the dominant variance term for any experiment
whose treatment lives *downstream* of retrieval.

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs=2, metavar="RUN", help="two results files of the SAME config")
    args = ap.parse_args()

    a, b = (load(x) for x in args.runs)
    pa, pb = provenance(a), provenance(b)
    print(f"run A provenance: {pa}\nrun B provenance: {pb}")
    if pa != pb:
        raise SystemExit(
            f"refusing to compare a {pa!r} run against a {pb!r} one: their candidate pools "
            "were produced differently, so the difference is not this benchmark's noise."
        )

    shared = sorted(set(a) & set(b))
    if len(shared) < 3:
        raise SystemExit(f"only {len(shared)} shared cases — not enough to estimate anything")

    d = [
        b[c]["reporadar_toppicks"]["net_value@2"] - a[c]["reporadar_toppicks"]["net_value@2"]
        for c in shared
    ]
    n = len(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    print(f"\n{n} shared cases, same configuration — every difference below is noise\n")
    print(
        f"  per-case delta   mean {statistics.mean(d):+.2f}   sd {sd:.2f}   "
        f"range [{min(d):+.0f}, {max(d):+.0f}]"
    )
    print(
        f"  identical on {sum(1 for x in d if x == 0)}/{n};  moved >=3 on "
        f"{sum(1 for x in d if abs(x) >= 3)}/{n}"
    )
    print(f"\n  SE of the {n}-case mean : {se:.2f}")
    print(f"  95% interval on a mean : +/-{1.96 * se:.2f}")
    print(f"  MINIMUM RESOLVABLE EFFECT (80% power, two-sided 0.05): {2.8 * se:.2f} net@2 per case")

    for key, label in (("reporadar_toppicks", "shown (Top Picks)"), ("reporadar_top10", "top-10")):
        vals = [j for c in shared if (j := jaccard(ids(a[c], key), ids(b[c], key))) is not None]
        if vals:
            print(f"  Jaccard overlap, {label:18}: {statistics.mean(vals):.3f}")

    print("\n  Sizing rule: an experiment aiming below the minimum resolvable effect cannot")
    print("  be answered by this instrument at this n. Freeze the pool (--rr-frozen-pool)")
    print("  when the treatment is downstream of retrieval, or accept an uninformative null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
