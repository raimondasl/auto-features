"""Refit the fine-scale map on Sonnet labels and see what the digest does. [NR-59]

`finescale.SLOPE`/`INTERCEPT` are a two-parameter logistic fitted on **judge-labelled
papers** — and the judge was GPT-5.5. `calibrate_finescale.py` says what rests on it:
*"Everything downstream of them — which papers clear P >= 2/3, and therefore the +4.55
headline — depends on that map still being located where it was fitted."*

So `finescale_p` is a calibrated estimate of **P(GPT calls this actionable)**, not
P(actionable). The threshold it is compared against is *not* fitted — 2/3 falls out of
net@2's own arithmetic, `3p - 2 > 0` — so the exposure is entirely in the map.

That matters because the two judges disagree, and **asymmetrically**: on the shipped digest
GPT calls 272/306 actionable (0.889) against Sonnet's 179/306 (0.585), while on Opus 5's
digest the two read 0.846 and 0.714. **Sonnet is 2.3x harsher on our picks than on the
comparator's** — which is the shape you would expect if our picks pass a filter fitted to
GPT and the comparator's do not. It is not the only explanation, and this probe is what
tells the explanations apart.

`calibrate_finescale.py` already names the honest counterfactual for the *repo* dimension:
*"a leave-one-repo-out refit, which never sees the repo it is scored on."* This is that
procedure applied to the *judge* dimension.

## Design

Three maps over one population — the 324 band papers of `second_judge_band.json`, 34 cases,
**244 shown and 80 withheld**, so both sides of the threshold are present and the range
restriction that spoils shown-only panels (NR-58, C-36) is structurally absent.

| map | fitted on | role |
|---|---|---|
| **shipped** | 219 papers, 22 repos, GPT labels — the frozen SLOPE/INTERCEPT | reproduction check |
| **GPT-refit** | these 324, GPT labels, leave-one-repo-out | control |
| **Sonnet-refit** | these 324, Sonnet labels, leave-one-repo-out | treatment |

The comparison that answers the question is **GPT-refit vs Sonnet-refit**: same papers, same
procedure, same fixed 2/3 threshold, *only the label differs*. The shipped map is not the
control — it was fitted on a different population, so a shipped-vs-Sonnet difference would
confound the label with the refit. It is here to check that refitting on GPT lands near where
the product already is; if it does not, this script is measuring itself.

**Leave-one-repo-out throughout.** A map that has seen a repo's own papers and is then scored
on them will look good under whichever label it was fitted to, which is the entire result
this probe could accidentally manufacture.

## Pre-registered, before any coefficient was fitted

* **Primary.** The fraction of the 324 band papers whose *display decision* (P >= 2/3) differs
  between the GPT-refit and Sonnet-refit maps.
  * **< 10% flip** — the map is not meaningfully judge-dependent. The sign flip NR-52 found
    comes from somewhere else, and the calibration is not where we are exposed.
  * **>= 25% flip** — the display rule is substantially an artifact of which judge labelled
    the calibration set, and it has to be refitted on something better than one model.
  * **10-25%** — material but partial; reported as such and neither claim is made.
* **Direction.** Sonnet is harsher, so a Sonnet-fitted map should put fewer papers over the
  line. A flip set that is symmetric, or that goes the other way, refutes the overfit story
  even if the rate is high.
* **Reproduction check, first and blocking.** The GPT-refit map must agree with the shipped
  map's recorded decisions on **>= 90%** of these papers. Below that, the refit procedure and
  the product are not the same operation and no comparison below is worth reading.
* **Prediction.** 15-30% flip, overwhelmingly in the show -> withhold direction. The two
  judges' base rates on this band differ by 0.31 (0.918 shown-rate under GPT against 0.570
  under Sonnet), and a logistic fitted to a lower base rate shifts its intercept down; a
  third of the band sits within one expectation-point of the line.

**What this cannot show.** Which map is *right*. Sonnet has no better claim to ground truth
than GPT — NR-56/57 could not separate them against adoption, the only model-free anchor, and
the difference missed its bar at n = 35. This measures **how much of the digest is a property
of the labelling judge**, which is a prior question and a cheaper one.

    uv run python evals/judge_refit.py            # $0, no LLM calls
    uv run python evals/judge_refit.py --report   # $0, re-read the artifact
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from reporadar.finescale import INTERCEPT, SHOW_THRESHOLD, SLOPE  # noqa: E402

BAND = EVALS / ".work" / "second_judge_band.json"
OUT = EVALS / "judge_refit.json"
ACTIONABLE = 2
REPRODUCTION_BAR = 0.90


def logistic(x: float, slope: float, intercept: float) -> float:
    return 1.0 / (1.0 + math.exp(-(slope * x + intercept)))


def fit_logistic(xs: list[float], ys: list[int], iters: int = 200) -> tuple[float, float]:
    """Two-parameter logistic by Newton-Raphson. Same functional form the product ships.

    Ridge-damped by a hair (1e-6 on the diagonal) so a separable subsample — which
    leave-one-repo-out will produce on the smaller repos — converges to a finite slope
    instead of running off. Stated because an unbounded coefficient would silently become an
    enormous flip count.
    """
    slope, intercept = 0.0, 0.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = logistic(x, slope, intercept)
            r = y - p
            w = max(p * (1 - p), 1e-9)
            g0 += r * x
            g1 += r
            h00 += w * x * x
            h01 += w * x
            h11 += w
        h00 += 1e-6
        h11 += 1e-6
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        d_slope = (h11 * g0 - h01 * g1) / det
        d_int = (h00 * g1 - h01 * g0) / det
        slope += d_slope
        intercept += d_int
        if abs(d_slope) < 1e-10 and abs(d_int) < 1e-10:
            break
    return slope, intercept


def loro_probabilities(rows: list[dict[str, Any]], label: str) -> list[float]:
    """P(actionable) per paper from a map that never saw that paper's repository.

    The discipline `calibrate_finescale` names for the repo dimension, applied here so the
    judge comparison cannot be won by a map memorising its own training cases.
    """
    out: list[float] = []
    for r in rows:
        train = [t for t in rows if t["case"] != r["case"]]
        xs = [float(t["finescale"]) for t in train]
        ys = [int(int(t[label]) >= ACTIONABLE) for t in train]
        slope, intercept = fit_logistic(xs, ys)
        out.append(logistic(float(r["finescale"]), slope, intercept))
    return out


def build() -> dict[str, Any]:
    if not BAND.is_file():
        raise SystemExit(f"{BAND} is not on disk; this probe reads the NR-52 band artifact")
    band = json.loads(BAND.read_text(encoding="utf-8"))
    rows = [
        r
        for r in band["rows"]
        if r.get("finescale") is not None
        and r.get("gpt_score") is not None
        and r.get("sonnet_score") is not None
    ]

    # The shipped map, replayed. Not refitted — this is the product's own decision.
    shipped_p = [logistic(float(r["finescale"]), SLOPE, INTERCEPT) for r in rows]
    shipped_show = [p >= SHOW_THRESHOLD for p in shipped_p]
    recorded_show = [bool(r["shown"]) for r in rows]
    replay_agreement = sum(a == b for a, b in zip(shipped_show, recorded_show, strict=True)) / len(
        rows
    )

    gpt_p = loro_probabilities(rows, "gpt_score")
    son_p = loro_probabilities(rows, "sonnet_score")
    gpt_show = [p >= SHOW_THRESHOLD for p in gpt_p]
    son_show = [p >= SHOW_THRESHOLD for p in son_p]

    reproduction = sum(a == b for a, b in zip(gpt_show, shipped_show, strict=True)) / len(rows)
    flips = [i for i in range(len(rows)) if gpt_show[i] != son_show[i]]
    to_withhold = [i for i in flips if gpt_show[i] and not son_show[i]]
    to_show = [i for i in flips if son_show[i] and not gpt_show[i]]

    # Whole-population coefficients, reported for readability. Every DECISION above is
    # leave-one-repo-out; these are what a shipped refit would carry.
    gpt_coef = fit_logistic(
        [float(r["finescale"]) for r in rows],
        [int(int(r["gpt_score"]) >= ACTIONABLE) for r in rows],
    )
    son_coef = fit_logistic(
        [float(r["finescale"]) for r in rows],
        [int(int(r["sonnet_score"]) >= ACTIONABLE) for r in rows],
    )

    def net2(show: list[bool], label: str) -> float:
        """net@2 over the band under one display rule and one judge, per case.

        The band is not a whole digest, so this is the band's CONTRIBUTION rather than a
        headline — said here because a number shaped like net@2 invites being read as one.
        """
        by_case: dict[str, int] = {}
        for r, s in zip(rows, show, strict=True):
            by_case.setdefault(r["case"], 0)
            if s:
                by_case[r["case"]] += 1 if int(r[label]) >= ACTIONABLE else -2
        return round(sum(by_case.values()) / len(by_case), 2)

    rate = len(flips) / len(rows)
    return {
        "_comment": (
            "NR-59: the fine-scale probability map refitted on Sonnet labels instead of GPT "
            "labels, leave-one-repo-out, at a FIXED 2/3 threshold. Derived by "
            "evals/judge_refit.py from evals/.work/second_judge_band.json (gitignored); "
            "pinned by tests/test_judge_refit.py. No LLM or judge calls. Measures how much "
            "of the digest is a property of the labelling judge -- not which judge is right, "
            "which NR-56/57 could not settle against adoption."
        ),
        "pre_registered": {
            "committed_before_any_coefficient_was_fitted": True,
            "primary": "fraction of band papers whose P>=2/3 decision differs between maps",
            "not_judge_dependent_below": 0.10,
            "substantially_an_artifact_at_or_above": 0.25,
            "reproduction_bar": REPRODUCTION_BAR,
            "direction": "Sonnet is harsher, so flips should be overwhelmingly show->withhold",
            "prediction": "15-30% flip, overwhelmingly show->withhold",
            "cannot_show": (
                "which map is right. Sonnet has no better claim to ground truth than GPT: "
                "NR-56/57 could not separate them against adoption and the difference missed "
                "its bar at n=35."
            ),
        },
        "population": {
            "n": len(rows),
            "n_cases": len({r["case"] for r in rows}),
            "n_shown_recorded": sum(recorded_show),
            "n_withheld_recorded": len(rows) - sum(recorded_show),
            "gpt_actionable_rate": round(
                sum(int(r["gpt_score"]) >= ACTIONABLE for r in rows) / len(rows), 4
            ),
            "sonnet_actionable_rate": round(
                sum(int(r["sonnet_score"]) >= ACTIONABLE for r in rows) / len(rows), 4
            ),
            "_comment": (
                "Both sides of the threshold are present -- 244 shown and 80 withheld -- so "
                "the range restriction that spoils shown-only panels (NR-58, C-36) is absent."
            ),
        },
        "reproduction": {
            "shipped_map_replays_recorded_decisions": round(replay_agreement, 4),
            "gpt_refit_agrees_with_shipped": round(reproduction, 4),
            "passes": reproduction >= REPRODUCTION_BAR,
            "_comment": (
                "The GPT refit must land near the product or the comparison below is between "
                "two things that are not the same operation. The shipped map is NOT the "
                "control -- it was fitted on a different population, so shipped-vs-Sonnet "
                "would confound the label with the refit."
            ),
        },
        "coefficients": {
            "shipped": {"slope": SLOPE, "intercept": INTERCEPT},
            "gpt_refit_all": {"slope": round(gpt_coef[0], 4), "intercept": round(gpt_coef[1], 4)},
            "sonnet_refit_all": {
                "slope": round(son_coef[0], 4),
                "intercept": round(son_coef[1], 4),
            },
            "_comment": "Reported for readability; every decision above is leave-one-repo-out.",
        },
        "flips": {
            "n": len(flips),
            "rate": round(rate, 4),
            "show_to_withhold": len(to_withhold),
            "withhold_to_show": len(to_show),
            "n_shown_gpt_refit": sum(gpt_show),
            "n_shown_sonnet_refit": sum(son_show),
            "examples": [
                {
                    "case": rows[i]["case"],
                    "arxiv_id": rows[i]["arxiv_id"],
                    "finescale": round(float(rows[i]["finescale"]), 2),
                    "p_gpt_map": round(gpt_p[i], 3),
                    "p_sonnet_map": round(son_p[i], 3),
                    "gpt_score": rows[i]["gpt_score"],
                    "sonnet_score": rows[i]["sonnet_score"],
                }
                for i in to_withhold[:8]
            ],
        },
        "band_net2_per_case": {
            "gpt_map_scored_by_gpt": net2(gpt_show, "gpt_score"),
            "gpt_map_scored_by_sonnet": net2(gpt_show, "sonnet_score"),
            "sonnet_map_scored_by_gpt": net2(son_show, "gpt_score"),
            "sonnet_map_scored_by_sonnet": net2(son_show, "sonnet_score"),
            "_comment": (
                "The band's CONTRIBUTION, not a headline -- it is one slice of the digest. "
                "The diagonal cells fit and score under the same label and are expected to "
                "flatter; the OFF-diagonal cells are what a reader should weigh."
            ),
        },
        "verdict": {
            "flip_rate": round(rate, 4),
            "judge_dependent": rate >= 0.10,
            "substantially_an_artifact": rate >= 0.25,
        },
    }


def show(art: dict[str, Any]) -> None:
    p, f, rep = art["population"], art["flips"], art["reproduction"]
    print(
        f"band: {p['n']} papers, {p['n_cases']} cases, {p['n_shown_recorded']} shown / "
        f"{p['n_withheld_recorded']} withheld"
    )
    print(
        f"actionable rate: GPT {p['gpt_actionable_rate']:.3f}   "
        f"Sonnet {p['sonnet_actionable_rate']:.3f}"
    )
    print(
        f"reproduction: GPT-refit agrees with shipped on {rep['gpt_refit_agrees_with_shipped']:.3f}"
        f"  (bar {art['pre_registered']['reproduction_bar']}) -> {rep['passes']}"
    )
    c = art["coefficients"]
    print(
        f"coefficients: shipped {c['shipped']['slope']:.3f}/{c['shipped']['intercept']:.3f}  "
        f"GPT {c['gpt_refit_all']['slope']:.3f}/{c['gpt_refit_all']['intercept']:.3f}  "
        f"Sonnet {c['sonnet_refit_all']['slope']:.3f}/{c['sonnet_refit_all']['intercept']:.3f}"
    )
    print(
        f"shown under each map: GPT-refit {f['n_shown_gpt_refit']}, "
        f"Sonnet-refit {f['n_shown_sonnet_refit']}"
    )
    print(
        f"FLIPS: {f['n']}/{p['n']} = {f['rate']:.1%}   "
        f"show->withhold {f['show_to_withhold']}, withhold->show {f['withhold_to_show']}"
    )
    n = art["band_net2_per_case"]
    print(f"{'band net@2/case':<22}{'by GPT':>9}{'by Sonnet':>11}")
    print(f"{'GPT-fit map':<22}{n['gpt_map_scored_by_gpt']:>9}{n['gpt_map_scored_by_sonnet']:>11}")
    print(
        f"{'Sonnet-fit map':<22}{n['sonnet_map_scored_by_gpt']:>9}"
        f"{n['sonnet_map_scored_by_sonnet']:>11}"
    )
    print()
    print("judge-dependent:", art["verdict"]["judge_dependent"])
    print("substantially an artifact:", art["verdict"]["substantially_an_artifact"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Refit the finescale map on Sonnet labels. $0.")
    ap.add_argument("--report", action="store_true", help="re-read the committed artifact")
    args = ap.parse_args()
    art = json.loads(OUT.read_text(encoding="utf-8")) if args.report else build()
    if not args.report:
        OUT.write_text(json.dumps(art, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT.name}")
    show(art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
