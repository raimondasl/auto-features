"""Dose-response report for the thin-docs ablation (`--rr-ablate-docs`).

    uv run python evals/ablation_report.py \
        control=judge-gpt-5.5-A.json 1500=...B.json 300=...C.json 0=...D.json

Every case in the benchmark has a README of at least 1,639 characters against a
300-character prose budget — **none under 1,000, none under 300** — so no measurement in
this project has ever run in the regime RepoRadar's stated target user lives in: a
private codebase with almost no prose. `--rr-ablate-docs` builds that regime out of a
real case by capping the README and withholding `docs/`, while the judge keeps seeing the
real repository so ground truth does not degrade alongside the treatment.

**What the arms isolate.** At a budget of 300 or more the gate's prose block is
*identical* to the control's — both are `README[:300]` — so those arms move only the
derived keywords, the queries built from them, and the HyDE hypotheses. That separates
**retrieval degradation from prompt degradation**, which the earlier prose-budget arms
(0/300/2000/6000) could not do: they varied what the gate was told about a repository
whose documentation was abundant either way.

**The question is the failure mode, not the score.** A thin repository *should* score
near zero — there is less to go on, and net@2 gives abstention a defined value of 0. The
alarm is a *negative* score, which would mean the system is confidently recommending
papers it cannot justify. The mechanism to worry about is specific: keyword retrieval
starves loudly (few candidates, the gate abstains), but `hyde.search_index` is pure
top-k by Hamming distance with **no similarity floor**, so a generic hypothesis still
returns its full quota of confidently-ranked but distant papers. The gate is the only
thing between that and the digest.

Pre-registered, before the numbers were read:

* **Prediction** — net@2 decays toward 0 by *abstention*, with pooled precision holding
  at or above 0.85 at every budget.
* **Alarm** — pooled precision below 0.80 at any budget, or any arm whose mean net@2 is
  negative: the abstention stance is not holding and HyDE needs a distance floor before
  a thin-docs user can be pointed at it.
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

PRECISION_ALARM = 0.80
PRECISION_PREDICTED = 0.85


def sign_test(deltas: list[float]) -> tuple[int, int, int, float]:
    """Two-sided exact sign test; ties dropped from n, then reported separately."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - pos - neg
    n = pos + neg
    if n == 0:
        return pos, neg, ties, 1.0
    k = min(pos, neg)
    return pos, neg, ties, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def load_arm(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        p = RESULTS_DIR / path
    return {r["case"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def summarise(arm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = [r["reporadar_toppicks"] for r in arm.values()]
    shown = sum(m["n_returned"] for m in tp)
    actionable = sum(m["n_actionable"] for m in tp)
    nets = [m["net_value@2"] for m in tp]
    return {
        "cases": len(arm),
        "mean_net2": statistics.mean(nets) if nets else 0.0,
        "shown": shown,
        "actionable": actionable,
        # None, not 0.0: an arm that showed nothing has no precision, and printing 0.00
        # would read as "everything it showed was wrong" — the opposite of what happened.
        "precision": actionable / shown if shown else None,
        "abstained": sum(1 for m in tp if m["n_returned"] == 0),
        "net_negative": sum(1 for v in nets if v < 0),
        "mean_pool": statistics.mean([r["pool_size"] for r in arm.values()]) if arm else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", metavar="LABEL=FILE", help="e.g. control=judge-...json")
    ap.add_argument("--out", default="evals/.work/ablation.json")
    args = ap.parse_args()

    arms: dict[str, dict[str, Any]] = {}
    for spec in args.arms:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=FILE, got {spec!r}")
        arms[label] = load_arm(path)

    labels = list(arms)
    control = labels[0]
    # Every arm must cover the same cases, or the means below compare different repos.
    base_cases = set(arms[control])
    for label in labels[1:]:
        if set(arms[label]) != base_cases:
            missing = base_cases ^ set(arms[label])
            raise SystemExit(f"arm {label!r} does not cover the same cases: {sorted(missing)}")

    print("=" * 78)
    print(f"THIN-DOCS DOSE RESPONSE — {len(base_cases)} cases, control arm = {control!r}")
    print("=" * 78)
    header = f"{'arm':>9} {'net@2':>7} {'shown':>6} {'act':>5} {'prec':>6} {'abst':>5} {'neg':>4}"
    print(f"\n{header} {'pool':>6}")
    summaries = {}
    for label in labels:
        s = summaries[label] = summarise(arms[label])
        prec = f"{s['precision']:.3f}" if s["precision"] is not None else "  n/a"
        print(
            f"{label:>9} {s['mean_net2']:+7.2f} {s['shown']:6} {s['actionable']:5} "
            f"{prec:>6} {s['abstained']:5} {s['net_negative']:4} {s['mean_pool']:6.1f}"
        )

    print(f"\n{'case':11}" + "".join(f"{label:>10}" for label in labels))
    for case in sorted(base_cases):
        row = "".join(
            f"{arms[label][case]['reporadar_toppicks']['net_value@2']:>10.1f}" for label in labels
        )
        print(f"{case:11}{row}")

    print("\npaired against the control arm, same session:")
    for label in labels[1:]:
        deltas = [
            arms[label][c]["reporadar_toppicks"]["net_value@2"]
            - arms[control][c]["reporadar_toppicks"]["net_value@2"]
            for c in sorted(base_cases)
        ]
        pos, neg, ties, p = sign_test(deltas)
        print(
            f"  {label:>9}  {statistics.mean(deltas):+6.2f} net@2/case   "
            f"{pos}+/{neg}-/{ties}=  p = {p:.4f}"
        )

    print("\n" + "-" * 78)
    verdicts = []
    for label, s in summaries.items():
        if s["precision"] is not None and s["precision"] < PRECISION_ALARM:
            verdicts.append(f"ALARM: {label} precision {s['precision']:.3f} < {PRECISION_ALARM}")
        if s["mean_net2"] < 0:
            verdicts.append(f"ALARM: {label} mean net@2 {s['mean_net2']:+.2f} < 0")
    if verdicts:
        print("\n".join(verdicts))
        print("The abstention stance is NOT holding. HyDE's top-k has no similarity floor.")
    else:
        low = min(s["precision"] for s in summaries.values() if s["precision"] is not None)
        met = "MET" if low >= PRECISION_PREDICTED else "below prediction, above the alarm"
        print(f"No alarm fired. Lowest pooled precision {low:.3f} — prediction {met}.")
        print("Degradation is by abstention, not by confident junk.")

    Path(args.out).write_text(
        json.dumps({"control": control, "summaries": summaries}, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
