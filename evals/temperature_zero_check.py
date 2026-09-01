"""Did `temperature=0` actually make the gate deterministic? [NR-55]

The verification for the one-line change NR-54 licensed. Two fresh runs of the shipped config
against the **same frozen pool**, post-fix, compared the way NR-54 compared its pre-fix pair —
identical design, so the two numbers are directly readable against each other.

**The prediction, stated before the runs finished, was 37/37 byte-identical. It was wrong.**

| | pre-fix (NR-54) | post-fix |
|---|---|---|
| byte-identical cases | 10 / 37 | **31 / 37** |
| per-case net@2 sd | **1.44** | **0.37** |
| median digest Jaccard | 0.857 | **1.000** |
| mean net@2 | — | +5.49 / +5.51 |

Of the six cases that still differ, **two are ordering only** (`columnar`, `rag` — same papers,
same scores, different order), which net@2 cannot see. Four differ in composition: `db` and
`numerics` swap one paper for another at identical net@2, `thin-kv` and `thin-lang` each drop
one. So **33 of 37 are deterministic in everything net@2 reads**, and the residual per-case sd
is 0.37 against 1.44 before — a **3.9x reduction**.

**The residual is most likely below the API, not in our code.** Greedy decoding fixes the
sampling rule; it does not make a served model bit-reproducible, because batching and
floating-point non-associativity can still move logits between requests. That is a hypothesis
this probe does not test — it is offered as the likeliest reading of a 4-case residual, not as a
finding — and the alternative worth checking if anyone cares is a second stochastic element
somewhere downstream of the gate.

**The dividend NR-54 projected is essentially fully realised.** It computed that a *perfectly*
deterministic gate would tighten the paired half-width from 0.78 to 0.63. With the residual
0.37 folded back in, the arm's paired sd becomes `sqrt(1.95^2 + 0.37^2) = 1.99` and the
half-width **0.64**. The gap between "perfect" and "achieved" is 0.01 net@2, which is nothing.

And it changes nothing about the plan, exactly as NR-54 said it would not: ladder rungs run
+0.20 to +0.45 and remain under 0.64, so the bundle-only rule stands.

    uv run python evals/temperature_zero_check.py     # $0 to re-derive
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id  # noqa: E402

RES = EVALS / "results"
POST_A = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260901T062305Z.json"
POST_B = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260901T072059Z.json"
FROZEN = EVALS / "temperature_zero_check.json"

PRE_SD = 1.44  # NR-54, same design, pre-fix
NR47_HALF_WIDTH = 0.78  # the resolution this is trying to improve
NR54_IDEAL_HALF_WIDTH = 0.63  # what NR-54 projected for a perfectly deterministic gate


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
    return sum(1 if s >= 2 else -2 for _p, s in picks)


def main() -> int:
    a, b = load(POST_A), load(POST_B)
    cases = sorted(set(a) & set(b))
    identical = [c for c in cases if a[c] == b[c]]
    order_only, composition = [], []
    for c in cases:
        if a[c] == b[c]:
            continue
        sa, sb = {p for p, _ in a[c]}, {p for p, _ in b[c]}
        (order_only if sa == sb else composition).append(c)

    d = [float(net(b[c]) - net(a[c])) for c in cases]
    sd = st.stdev(d)
    jac = [
        len({p for p, _ in a[c]} & {p for p, _ in b[c]})
        / max(1, len({p for p, _ in a[c]} | {p for p, _ in b[c]}))
        for c in cases
    ]

    # NR-54's decomposition, with the measured residual folded back in.
    sd_arm = NR47_HALF_WIDTH * math.sqrt(len(cases)) / 1.96
    sd_treat = math.sqrt(max(0.0, sd_arm**2 - PRE_SD**2))
    sd_new = math.sqrt(sd_treat**2 + sd**2)
    hw_new = 1.96 * sd_new / math.sqrt(len(cases))

    out: dict[str, Any] = {
        "_comment": (
            "NR-55. Verification of the temperature=0 change, using NR-54's design so the two "
            "are directly comparable: two fresh runs of the shipped config against the SAME "
            "frozen pool. The prediction stated before the runs finished was 37/37 "
            "byte-identical and it was WRONG -- 31/37, with 2 more differing only in ordering. "
            "Derived by evals/temperature_zero_check.py; pinned by "
            "tests/test_temperature_zero_check.py."
        ),
        "prediction_was": {"byte_identical": 37, "of": 37, "held": False},
        "n_cases": len(cases),
        "byte_identical": len(identical),
        "order_only": sorted(order_only),
        "composition_differs": sorted(composition),
        "net2_relevant_determinism": len(cases) - len(composition),
        "per_case_sd": round(sd, 2),
        "mean_delta": round(st.mean(d), 2),
        "median_digest_jaccard": round(st.median(jac), 3),
        "mean_net2": [
            round(st.mean(net(a[c]) for c in cases), 2),
            round(st.mean(net(b[c]) for c in cases), 2),
        ],
        "vs_pre_fix": {
            "pre_sd": PRE_SD,
            "post_sd": round(sd, 2),
            "reduction_factor": round(PRE_SD / sd, 1) if sd else None,
            "pre_byte_identical": 10,
        },
        "residual": {
            "_comment": (
                "Offered as the likeliest reading of a 4-case residual, NOT as a finding this "
                "probe tests. Greedy decoding fixes the sampling rule; it does not make a "
                "served model bit-reproducible, because batching and floating-point "
                "non-associativity can move logits between requests. The alternative worth "
                "checking is a second stochastic element downstream of the gate."
            ),
            "likely_cause": "API-level nondeterminism below temperature",
            "cases": sorted(composition),
        },
        "dividend": {
            "half_width_before": NR47_HALF_WIDTH,
            "nr54_projected_ideal": NR54_IDEAL_HALF_WIDTH,
            "half_width_achieved": round(hw_new, 2),
            "gap_to_ideal": round(hw_new - NR54_IDEAL_HALF_WIDTH, 2),
            "ladder_rungs_still_unresolvable": bool(hw_new > 0.45),
        },
    }
    out["verdict"] = {
        "gate_is_near_deterministic": bool(sd < 0.5),
        "prediction_held": False,
        "dividend_essentially_realised": bool(abs(hw_new - NR54_IDEAL_HALF_WIDTH) <= 0.05),
        "changes_the_plan": False,
    }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"POST-FIX: two runs, identical frozen pool, temperature=0, {len(cases)} cases\n")
    print(f"  byte-identical        : {len(identical)}/{len(cases)}   (pre-fix: 10/37)")
    print(f"  order-only differences: {len(order_only)}  {sorted(order_only)}")
    print(f"  composition differs   : {len(composition)}  {sorted(composition)}")
    print(f"  net@2-relevant determinism: {len(cases) - len(composition)}/{len(cases)}")
    print(
        f"\n  per-case net@2 sd     : {sd:.2f}   (pre-fix {PRE_SD})  -> {PRE_SD / sd:.1f}x tighter"
    )
    print(f"  median digest Jaccard : {st.median(jac):.3f}   (pre-fix 0.857)")
    print(f"  mean net@2            : {out['mean_net2'][0]:+.2f} / {out['mean_net2'][1]:+.2f}")
    print(f"\n  half-width before     : {NR47_HALF_WIDTH}")
    print(f"  NR-54 projected ideal : {NR54_IDEAL_HALF_WIDTH}")
    print(f"  achieved              : {hw_new:.2f}")
    print(f"\nprediction was 37/37 byte-identical: WRONG ({len(identical)}/37)")
    print(f"wrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
