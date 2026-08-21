"""What shape are the gate's scores, per case and per domain? ($0, offline, judge-free.)

    uv run python evals/gate_shape.py

**This exists to check §17.4, and it refutes it.** §17.4 observed that fourteen of linter's
fifteen ranked papers carried the single gate score 1 and called that "degenerate on its own
terms, whatever any judge thinks" — offered as the part of the linter observation that survives
after §17.2 retracted the rest. It cited two cases as contrast (http at 13/2, CHGNet spread
across 2 and 3) and never asked what the other thirty-four look like.

They look like linter. The gate puts a median **73%** of a ranked window into one bucket, eight
of thirty-seven cases sit at or above linter's 93%, and two cases are at **100%** — `rl` and
`thin-kv`, which are among the run's better results. Concentration is the gate's normal
behaviour, not a defect signature, so "93% in one bucket" licenses nothing on its own.

What distinguishes linter is not how sharp its distribution is but **where the mode sits**:
14 of 15 at score 1, below the admit threshold, where `rl`'s 15 of 15 sit at score 2, above it.
That is a restatement of "linter returned nothing", not independent evidence about the gate.

The lesson, recorded because it cost a second retraction on the same paragraph: §17.2 removed
the judge from a claim and §17.4 assumed that made it safe. It did not. A judge-free claim still
needs a control, and the control for "this distribution is degenerate" is the other thirty-six
distributions. Removing a dependency changes which control a claim needs; it does not remove
the need for one.

**What does survive is a different statistic entirely** — how often the gate reaches for each
score, which needs no labels: it emits its top score on 20.0% of scientific-software papers
against 8.0% of ML/CS ones (Fisher p = 7.7e-05), and never once emits its bottom score on
scientific software (0/180 against 31/375). That is the judge-free companion to §16.5, which
measured the same asymmetry through the judge.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))

from finescale_domains import DEFAULT_LEGACY, DEFAULT_SCI  # noqa: E402
from label_pool import fisher_exact  # noqa: E402
from run_judge_eval import RESULTS_DIR  # noqa: E402

SCORES = (0, 1, 2, 3)
FOCUS = "linter"  # the case §17.4 was written about


def concentration(hist: collections.Counter[Any]) -> tuple[float, float, int]:
    """Largest bucket share, entropy normalised over the four available scores, and k.

    Normalised entropy uses log base 4 so a flat distribution over all four scores scores 1.0
    and a single-bucket one scores 0.0, independent of how many scores the case happens to use.
    """
    n = sum(hist.values())
    top = max(hist.values()) / n
    ent = -sum((v / n) * math.log(v / n, len(SCORES)) for v in hist.values())
    return top, ent, len(hist)


def load(path: Path, population: str) -> list[dict[str, Any]]:
    rows = []
    for rec in json.loads(path.read_text(encoding="utf-8")):
        hist = collections.Counter(p.get("llm_score") for p in rec["returned"]["reporadar_top10"])
        top, ent, k = concentration(hist)
        rows.append(
            {
                "case": rec["case"],
                "population": population,
                "n": sum(hist.values()),
                "hist": {s: hist.get(s, 0) for s in SCORES},
                "top_share": top,
                "entropy": ent,
                "k": k,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sci", default=DEFAULT_SCI)
    ap.add_argument("--legacy", default=DEFAULT_LEGACY)
    args = ap.parse_args()

    rows = []
    for name, population in ((args.sci, "sci"), (args.legacy, "legacy")):
        path = Path(name) if Path(name).is_file() else RESULTS_DIR / name
        rows += load(path, population)
    rows.sort(key=lambda r: -r["top_share"])

    print("=" * 78)
    print("GATE SCORE CONCENTRATION per case, judge-free, most concentrated first")
    print("=" * 78)
    print(f"  {'case':16} {'pop':6} {'n':>3} {'top':>6} {'entropy':>8} {'k':>2}  histogram 0/1/2/3")
    for r in rows:
        h = r["hist"]
        mark = "   <-- §17.4" if r["case"] == FOCUS else ""
        print(
            f"  {r['case']:16} {r['population']:6} {r['n']:3d} {r['top_share']:6.0%}"
            f" {r['entropy']:8.3f} {r['k']:2d}  {h[0]:2d}/{h[1]:2d}/{h[2]:2d}/{h[3]:2d}{mark}"
        )

    shares = sorted(r["top_share"] for r in rows)
    focus = next((r for r in rows if r["case"] == FOCUS), None)
    print(
        f"\n  median top-bucket share: {shares[len(shares) // 2]:.0%}"
        f"   min {shares[0]:.0%}   max {shares[-1]:.0%}"
    )
    if focus is not None:
        at_or_above = sum(1 for s in shares if s >= focus["top_share"] - 1e-9)
        perfect = [r["case"] for r in rows if r["k"] == 1]
        print(
            f"  {FOCUS} is at {focus['top_share']:.0%}; {at_or_above} of {len(rows)} cases "
            f"are at or above it."
        )
        print(f"  cases using exactly ONE score: {perfect or 'none'}")
        print(
            f"\n  => §17.4's premise fails. Concentration this high is ordinary, and the two\n"
            f"     most concentrated cases are not failures. What is unusual about {FOCUS} is\n"
            f"     that its mode sits BELOW the admit threshold, which is the same fact as\n"
            f"     'it returned nothing' rather than evidence for it."
        )

    print("\n" + "=" * 78)
    print("SCORE EMISSION RATE by population, judge-free")
    print("=" * 78)
    totals = {
        p: collections.Counter(
            {s: sum(r["hist"][s] for r in rows if r["population"] == p) for s in SCORES}
        )
        for p in ("sci", "legacy")
    }
    n_sci, n_leg = sum(totals["sci"].values()), sum(totals["legacy"].values())
    print(f"  {'score':>5} {'scientific-12':>20} {'legacy-25':>20} {'Fisher p':>12}")
    for s in SCORES:
        a, b = totals["sci"][s], n_sci - totals["sci"][s]
        c, d = totals["legacy"][s], n_leg - totals["legacy"][s]
        p = fisher_exact(a, b, c, d)
        print(f"  {s:5d} {a:8d}/{n_sci} {a / n_sci:6.1%} {c:8d}/{n_leg} {c / n_leg:6.1%} {p:12.2e}")
    print(
        "\n  No judge appears anywhere above. This says the gate REACHES for its top score more\n"
        "  often on scientific software and never reaches for its bottom one — a fact about the\n"
        "  gate's behaviour, not about whether it was right. §16.5 measured the same asymmetry\n"
        "  through the judge (score-3 non-actionable 31% vs 7%); the two agree without sharing\n"
        "  a dependency, which is the strongest form this project's evidence takes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
