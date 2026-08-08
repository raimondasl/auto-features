"""The head-to-head: RepoRadar + the fine-scale rescore vs the Opus baseline, all 22 repos.

Costs nothing and calls nothing. The Opus baseline's picks and the judge's verdicts on them
are already inside the frozen run file, alongside RepoRadar's, so the comparison is a replay
rather than a new benchmark — which also means the two systems are scored on the same
candidates, by the same judge, in the same session.

The probabilities are **leave-one-repo-out**: for each held-out repo the score→probability
map is fitted on the other 21 and applied to that one. That is the honest estimate of what
the shipped map does on a repo it has not seen, and it deliberately uses the SAME model
family as `reporadar.finescale` (a plain unregularised logistic on the raw 0-9 expectation).
Fitting a differently-regularised map and reporting its number is exactly the error the
"Correction" section of RESULTS.md documents: `exp_features.loro_fit` selects L2 strength by
AUC, which is rank-only and therefore blind to where P crosses the 2/3 threshold.

    uv run python evals/compare_finescale_baseline.py
    uv run python evals/compare_finescale_baseline.py --testbed a300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402


def loro_probabilities(
    bands: dict[str, tb.CaseBand], scored: dict[str, dict[str, dict]]
) -> dict[str, dict[str, float]]:
    """{case: {paper_id: P(actionable)}}, each case predicted by a map that never saw it."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rows = [
        (case, p.id, scored.get(case, {}).get(p.id, {}).get("exp09"), int(p.actionable))
        for case, band in bands.items()
        for p in band.papers
    ]
    rows = [r for r in rows if r[2] is not None]
    out: dict[str, dict[str, float]] = {}
    for held in sorted(bands):
        train = [(x, y) for case, _, x, y in rows if case != held]
        if len({y for _, y in train}) < 2:
            continue
        model = LogisticRegression(max_iter=1000).fit(
            np.array([[x] for x, _ in train]), np.array([y for _, y in train])
        )
        for case, pid, x, _ in rows:
            if case == held:
                out.setdefault(case, {})[pid] = float(model.predict_proba([[x]])[0, 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testbed", default="a", choices=["a", "a300"])
    args = ap.parse_args()

    bands = tb.load_testbed_a() if args.testbed == "a" else tb.load_testbed_a300()
    path = tb.EXP / f"finescale_{args.testbed}.json"
    if not path.is_file():
        raise SystemExit(f"no {path} — run `uv run python evals/exp_finescale.py` first")
    scored = json.loads(path.read_text(encoding="utf-8"))["scored"]
    run_file = tb.POOL50 if args.testbed == "a" else tb.POOL300
    runs = {e["case"]: e for e in json.loads(run_file.read_text(encoding="utf-8"))}
    for case, rerun in tb.POOL300_RERUNS.items():
        if args.testbed == "a300":
            entries = json.loads(rerun.read_text(encoding="utf-8"))
            runs[case] = next(e for e in entries if e["case"] == case)

    probs = loro_probabilities(bands, scored)
    cases = sorted(bands)

    # ASCII only: the default Windows console codepage is cp1252 and cannot encode "Δ".
    print(f"{'case':11} {'show-all':>9} {'+finescale':>11} {'Opus':>6} {'vs Opus':>10}")
    totals = dict.fromkeys(("show", "fine", "base"), 0.0)
    deltas_fine, deltas_show = [], []
    shown = good = 0
    for case in cases:
        band = bands[case]
        per = probs.get(case, {})
        show = tb.net2([p.judge for p in band.admitted])
        fine = tb.policy_net(band, per)
        base = runs[case]["baseline"]["net_value@2"]
        digest = band.gate3 + [p for p in band.band2 if per.get(p.id, -1.0) >= tb.SHOW_THRESHOLD]
        shown += len(digest)
        good += sum(1 for p in digest if p.actionable)
        totals["show"] += show
        totals["fine"] += fine
        totals["base"] += base
        deltas_fine.append(fine - base)
        deltas_show.append(show - base)
        print(f"{case:11} {show:+9.1f} {fine:+11.1f} {base:+6.1f} {fine - base:+10.1f}")

    n = len(cases)
    print(
        f"\n{'MEAN':11} {totals['show'] / n:+9.2f} {totals['fine'] / n:+11.2f} "
        f"{totals['base'] / n:+6.2f} {sum(deltas_fine) / n:+10.2f}"
    )
    print(
        f"\npaired vs Opus : show-all {sum(deltas_show) / n:+.2f}, "
        f"+finescale {sum(deltas_fine) / n:+.2f}"
    )
    print(f"sign test      : {tb.sign_test(deltas_fine)}")
    print(f"digest         : {shown} shown, {good} actionable, precision {good / shown:.2f}")
    base_shown = sum(runs[c]["baseline"]["n_returned"] for c in cases)
    base_good = sum(runs[c]["baseline"]["n_actionable"] for c in cases)
    print(
        f"Opus           : {base_shown} shown, {base_good} actionable, "
        f"precision {base_good / base_shown:.2f}"
    )
    print(
        f"net-negative   : finescale "
        f"{[c for c in cases if tb.policy_net(bands[c], probs.get(c, {})) < 0]}, "
        f"Opus {[c for c in cases if runs[c]['baseline']['net_value@2'] < 0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
