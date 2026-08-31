"""Item 12's paid arm: PRF-HyDE does not move net@2. The item closes. [NR-51]

The arm NR-50 licensed, ~$15. Control is the shipped arm already on disk; treatment is
identical in every flag except a pinned hypothesis file carrying **round 1 union round 2** —
8 abstracts per case, searched through the shipped `hyde.discover` at `top_k` 100, so the
pool is collected by the product's own code and the fingerprint is honest about being a
different pool.

**The pre-registered kill fires. |−0.19| < 0.78.**

| arm | net@2 | digest/case | precision |
|---|---|---|---|
| control (ships) | **+5.51** | 8.3 | 0.889 |
| treatment (PRF) | +5.32 | 8.5 | 0.876 |

    primary, all 37   -0.19   CI [-0.84, +0.43]   9w/8l/20t   p = 1.0000
    secondary, 33     -0.21   CI [-0.97, +0.52]   9w/8l/16t   p = 1.0000

**And the free prior had the sign wrong.** NR-50 read the judge cache over *window* papers and
got Δp = **+0.054**, favouring round 2, while flagging it as weak — 61% of entering and 73% of
displaced papers were void, and the judged subset is selected by having been shown. Measured
properly in the digest, the churn is **85 added at 0.882 precision displacing 77 at 0.935:
Δp = −0.053**. Same magnitude, opposite sign. The caution was right; the estimate was not, and
that is the more useful half of this result.

**The four no-round-2 cases came back as exact ties with identical picks**, which is the
arm's own validation: where the shipped run showed nothing to feed on, the treatment *is* the
control, byte for byte, and nothing was fabricated to fill the gap.

**Two cases had to be repaired**, both for the reason NR-47 documented. `compiler` and
`numerics` lost HyDE entirely to an arXiv 429 and 503 after 10 attempts and ~930 s each,
collecting keyword-only pools (519 → 211, 589 → 202). Paired against a control that *has*
HyDE they would have measured "HyDE existing at all" rather than "round 2 added" — the largest
confound available in a 37-case paired test. Re-collected at identical flags and spliced in
(519 → 843, 589 → 957), rather than dropped.

**What closes.** Item 12 was the last open evidence-led lead. Reach was null at two budget
points (NR-49); the rank probe licensed this arm because round 2 takes a fifth of the gate
window (NR-50) and then showed that share *rising* with depth, meaning marginal placement; and
the arm itself returns a wash. Retrieval width (NR-47), gate depth (NR-48) and now iterative
retrieval have each been measured and none pays.

    uv run python evals/freeze_prf_arm.py     # $0 to re-derive; the arm itself cost ~$15
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from band_testbeds import sign_test  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

RES = EVALS / "results"
CONTROL = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T034455Z.json"
CONTROL_REPAIR = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T075622Z.json"
TREAT = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260831T045338Z.json"
TREAT_REPAIRS = {
    "numerics": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260831T045849Z.json",
    "compiler": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260831T050239Z.json",
}
NO_ROUND2 = ("cli", "http", "linter", "webdev")
BAR = 0.78  # NR-47's bootstrap half-width at n = 37; registered before the run
FROZEN = EVALS / "prf_arm.json"


def points(score: int) -> int:
    return 1 if int(score) >= 2 else -2


def load(name: str) -> dict[str, list[tuple[str, int]]]:
    run = json.loads((RES / name).read_text(encoding="utf-8"))
    return {
        e["case"]: [
            (dedup_id(str(p["arxiv_id"])), int(p["judge_score"]))
            for p in e["returned"]["reporadar_toppicks"]
        ]
        for e in run
    }


def net(picks: list[tuple[str, int]]) -> int:
    return sum(points(s) for _p, s in picks)


def arm(label: str, a: dict, cases: list[str]) -> dict[str, Any]:
    flat = [s for c in cases for _p, s in a[c]]
    return {
        "label": label,
        "mean_net2": round(st.mean(net(a[c]) for c in cases), 2),
        "digest_per_case": round(sum(len(a[c]) for c in cases) / len(cases), 1),
        "precision": round(sum(1 for s in flat if s >= 2) / len(flat), 3),
    }


def compare(ctrl: dict, treat: dict, cases: list[str]) -> dict[str, Any]:
    d = [float(net(treat[c]) - net(ctrl[c])) for c in cases]
    lo, hi = paired_bootstrap(d)
    s = sign_test(d)
    return {
        "n_cases": len(cases),
        "paired_delta": round(st.mean(d), 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "wins": s["pos"],
        "losses": s["neg"],
        "ties": s["ties"],
        "sign_p": round(s["p"], 4),
    }


def main() -> int:
    ctrl = load(CONTROL)
    ctrl["bio-mdtraj"] = load(CONTROL_REPAIR)["bio-mdtraj"]
    treat = load(TREAT)
    for case, f in TREAT_REPAIRS.items():
        treat[case] = load(f)[case]

    cases = sorted(set(ctrl) & set(treat))
    with_r2 = [c for c in cases if c not in NO_ROUND2]

    added: list[tuple[str, int]] = []
    dropped: list[tuple[str, int]] = []
    for c in cases:
        ci, ti = dict(ctrl[c]), dict(treat[c])
        added += [(p, s) for p, s in ti.items() if p not in ci]
        dropped += [(p, s) for p, s in ci.items() if p not in ti]

    def prec(v: list[tuple[str, int]]) -> float:
        return round(sum(1 for _p, s in v if s >= 2) / len(v), 3) if v else float("nan")

    primary = compare(ctrl, treat, cases)
    out: dict[str, Any] = {
        "_comment": (
            "NR-51 / item 12's paid arm. Derived by evals/freeze_prf_arm.py; pinned by "
            "tests/test_prf_arm.py. The PRE-REGISTERED KILL FIRES: |-0.19| < 0.78, so this is "
            "a NULL and item 12 closes for good. Treatment differs from control only in the "
            "pinned hypothesis file, which carries round 1 UNION round 2 (8 abstracts/case) "
            "searched through the shipped hyde.discover at top_k 100."
        ),
        "pre_registered": {
            "bar": BAR,
            "bar_is": "NR-47's paired-bootstrap half-width at n = 37",
            "criterion": "|paired delta| >= bar resolves; below it is a null and item 12 closes",
            "written_before_the_run": True,
            "expected_outcome_was_null": (
                "Recorded in advance so the result could not be re-read. Reach was null at two "
                "budget points; the 20.61% window share licensed the arm at a GENEROUS dp=0.2; "
                "the measured dp implied +0.28/case; the top-15 correction implied +0.23."
            ),
        },
        "arms": {
            "control": arm("shipped", ctrl, cases),
            "treatment": arm("round 1 union round 2", treat, cases),
        },
        "primary_all_37": primary,
        "secondary_33_with_round2": compare(ctrl, treat, with_r2),
        "cohorts": {
            label: round(st.mean(net(treat[c]) - net(ctrl[c]) for c in sel), 2)
            for label, sel in (
                ("core25", [c for c in cases if not c.startswith(("bio-", "mat-"))]),
                ("bio6", [c for c in cases if c.startswith("bio-")]),
                ("matsci6", [c for c in cases if c.startswith("mat-")]),
            )
        },
        "digest_churn": {
            "_comment": (
                "NR-48's diagnostic, and the number that overturns NR-50's free prior. That "
                "prior read the judge cache over WINDOW papers and got dp = +0.054 favouring "
                "round 2, flagged as weak because 61%/73% of the two sides were void and the "
                "judged subset is selected by having been shown. Measured in the digest the "
                "sign INVERTS: what PRF adds is worse than what it displaces."
            ),
            "added": {"n": len(added), "precision": prec(added)},
            "dropped": {"n": len(dropped), "precision": prec(dropped)},
            "observed_dp": round(prec(added) - prec(dropped), 3),
            "nr50_prior_dp": 0.054,
            "prior_had_the_sign_wrong": True,
        },
        "no_round2_cases": {
            "_comment": (
                "The arm's own validation. These four had nothing for round 2 to feed on, so "
                "the treatment IS the control there. They were never padded with a fresh draw "
                "-- that would have fabricated a treatment -- and they come back as exact ties "
                "with byte-identical picks, which is what confirms the two arms differ in one "
                "thing only."
            ),
            "cases": list(NO_ROUND2),
            "all_exact_ties": all(ctrl[c] == treat[c] for c in NO_ROUND2 if c in cases),
        },
        "repaired_cases": {
            "_comment": (
                "Both lost HyDE entirely to arXiv throttling (429 and 503) after 10 attempts "
                "and ~930s each, collecting keyword-only pools. Paired against a control that "
                "HAS HyDE they would have measured 'HyDE existing at all' rather than 'round 2 "
                "added' -- the same confound NR-47 repaired for bio-mdtraj. Re-collected at "
                "identical flags and spliced in rather than dropped."
            ),
            "compiler": {"degraded_pool": 211, "shipped_pool": 519, "repaired_pool": 843},
            "numerics": {"degraded_pool": 202, "shipped_pool": 589, "repaired_pool": 957},
        },
        "per_case": {c: {"control": net(ctrl[c]), "treatment": net(treat[c])} for c in cases},
    }
    out["verdict"] = {
        "killed": abs(primary["paired_delta"]) < BAR,
        "resolves": abs(primary["paired_delta"]) >= BAR,
        "direction_closed": True,
        "closing_note": (
            "A wash against what ships, at an extra ~250 candidates per case. Item 12 was the "
            "last open evidence-led lead. Retrieval width (NR-47), gate depth (NR-48) and now "
            "iterative retrieval have each been measured, and none pays."
        ),
    }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: out[k] for k in ("arms", "primary_all_37", "digest_churn", "verdict")}, indent=1
        )
    )
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
