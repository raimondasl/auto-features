"""Items 10 and 13: both pool-volume levers are spent. [NR-47, NR-48]

Three arms over the same 37 cases, one pinned hypothesis set throughout:

    A  ships    hyde.top_k=100    gate_depth=50    net@2 +5.51   8.3 papers/case   prec 0.889
    B  wide     hyde.top_k=1000   gate_depth=50    net@2 +4.73   7.4              prec 0.880
    C  deep     hyde.top_k=1000   gate_depth=150   net@2 +5.32   9.0              prec 0.864

**NR-48 / item 13 is the second half and it closes the direction.** NR-47 found that widening
retrieval SHRANK the digest -- a 5.9x pool met a gate reading a fixed 50 of it -- and its
diagnostic said the added papers were fine (0.882 against 0.878), so the fault looked like the
window rather than the material. Item 13 tested that directly, reusing B's pools because
`rr_pool` is not a POOL_FLAG.

**Pre-registered: the digest must return to at least 8.3/case AND net@2 to at least +5.51.**
The digest recovered -- 9.0, past the shipped size. **net@2 did not: +5.32.** That was written
in advance as the kill: *if the digest recovers but net@2 does not, the papers a deeper window
admits are worse than the ones they displace, and the whole direction closes.*

It is exactly what the deeper window did: **99 papers admitted at precision 0.859, displacing
41 at 0.951.** Ranks 50-150 of the ranked pool are genuinely thinner material. Those admissions
are still above net@2's 2/3 break-even, so depth recovers +0.59 of NR-47's -0.78 -- but not all
of it, and the end state is a **wash against what ships**: -0.19, CI [-1.14, +0.78], 14w/15l/8t,
at 5.9x the pool and 3x the gate calls. No gain, real cost.

**So the pipeline is at its efficient frontier for pool volume.** Retrieval width and gate depth
were the two ways to feed it more, and neither pays. That is a stronger statement than either
arm alone, and it is what NR-11's warning generalises to now: not "a wider pool met a near-binary
gate", but "more candidates do not help this system at any depth we can afford to judge".

---

Item 10 stage 2 [NR-47]: the wider HyDE cut costs net@2, and the reason is not the papers.

The paid arm NR-46's stage 1 licensed. Two same-day arms over 37 cases, everything fixed but
`hyde.top_k` -- 100 against 1000 -- sharing **one pinned hypothesis set**, because NR-46
measured a hypothesis redraw at +0.058 witness reach and two arms drawing their own could not
have told the cut from the draw. `hyde.discover` gained an optional `hypotheses=` for exactly
this; the shipped path is unchanged.

**The kill condition fires.** Stage 2 was pre-registered as "net@2 must not fall".

    all 37    control +5.51   treatment +4.73   paired -0.78  CI [-1.59, -0.03]  13w/17l/7t
    core 25           +5.92             +5.40          -0.52
    bio 6             +4.67             +4.17          -0.50
    matsci 6          +4.67             +2.50          -2.17

The bootstrap interval sits (barely) below zero; the sign test does not resolve at p = 0.58.
Read together: a small, consistent loss, and the item closes on it.

**Stage 1 was right about reach and that was not enough.** Reach doubled, exactly as simulated.
It bought nothing, which is the fourth time this project has measured a pool expansion as a
wash or worse -- NR-11, P4, and now this -- and the first time the mechanism is visible in the
same run.

**The diagnostic, which is worth more than the headline.** Of what changed in the digests:

    kept     164 papers   precision 0.878
    added    110 papers   precision 0.882    net@2 contribution +1.92/case
    dropped  142 papers   precision 0.901    net@2 forgone      -2.70/case
    digest size 8.3 -> 7.4 per case, from a pool 5.9x larger

**The papers the wider cut adds are fine.** At 0.882 they are indistinguishable from the 0.878
that were already there. The loss is that a 5.9x larger pool produced a **smaller digest**, and
splitting the -0.78 exactly says so: **-0.609 from showing 32 fewer papers (78%)** and -0.175
from the ones it did show being slightly worse. The two terms sum to the delta by construction.

A candidate set six times larger meets a gate that still reads `gate_depth` 50 of it, so the
extra reach arrives as dilution and fewer admissions -- NR-11's mechanism, observed in the same
run rather than inferred from a later one.

That is the "the gate never saw them" branch of the fork stage 1 pre-registered, and it makes
the follow-up specific rather than hopeful: the papers are good, the window is not wide enough
to let them be judged. It does **not** license shipping a wider cut, and this artifact does not.

**One case needed repair before any of this was comparable.** `bio-mdtraj`'s control collection
hit arXiv HTTP 429 after 10 attempts and 930s of throttle-waiting, so its pool fell back to
keyword-only -- zero HyDE candidates. Paired against a treatment arm that had HyDE, its delta
would have measured *HyDE existing at all* rather than the cut. The harness reported it loudly
("this arm is NOT a clean HyDE measurement"), the case was re-collected at identical flags once
the throttle cleared, and the repair is spliced in here. The alternative -- dropping it -- was
available and is worse: 36 of 37 with an unexplained gap invites the reader to wonder which.
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
FROZEN = EVALS / "hyde_cut_arm.json"
ARMS = {
    "A_ships_k100_d50": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T034455Z.json",
    "B_wide_k1000_d50": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T075259Z.json",
    # NR-48 / item 13: gate depth 50 -> 150 on the SAME pools B collected. `rr_pool` is not a
    # POOL_FLAG, so the 5.9x collection is reused rather than re-bought.
    "C_deep_k1000_d150": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T171257Z.json",
}
GATE_DEPTH = {"A_ships_k100_d50": 50, "B_wide_k1000_d50": 50, "C_deep_k1000_d150": 150}
REPAIR = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T075622Z.json"
REPAIRED_CASE = "bio-mdtraj"
POOLS = {
    "A_ships_k100_d50": "pool-cut100",
    "B_wide_k1000_d50": "pool-cut1000",
    "C_deep_k1000_d150": "pool-cut1000",
}
COHORTS = {
    "core25": lambda c: not c.startswith(("bio-", "mat-")),
    "bio6": lambda c: c.startswith("bio-"),
    "matsci6": lambda c: c.startswith("mat-"),
    "all37": lambda _c: True,
}


def pt(score: int) -> int:
    return 1 if int(score) >= 2 else -2


def net(picks) -> int:
    return sum(pt(s) for _p, s in picks)


def load(fname: str) -> dict[str, list[tuple[str, int]]]:
    run = json.loads((RES / fname).read_text(encoding="utf-8"))
    return {
        e["case"]: [
            (dedup_id(str(p["arxiv_id"])), int(p["judge_score"]))
            for p in e["returned"]["reporadar_toppicks"]
        ]
        for e in run
    }


def _any_pool_config(name: str) -> dict[str, Any]:
    d = EVALS / ".work" / POOLS[name]
    f = next(iter(sorted(d.glob("*.json"))))
    return json.loads(f.read_text(encoding="utf-8"))["pool_config"]


def recorded_top_k(name: str) -> int:
    return int(_any_pool_config(name)["rr_hyde_top_k"])


def recorded_hyp_file(name: str) -> str | None:
    return _any_pool_config(name).get("rr_hyde_hypotheses_file")


def pool_sizes(name: str) -> dict[str, int]:
    d = EVALS / ".work" / POOLS[name]
    return {f.stem: json.loads(f.read_text(encoding="utf-8"))["n"] for f in d.glob("*.json")}


def main() -> int:
    picks = {name: load(f) for name, f in ARMS.items()}
    picks["A_ships_k100_d50"][REPAIRED_CASE] = load(REPAIR)[REPAIRED_CASE]
    ctrl, treat = picks["A_ships_k100_d50"], picks["B_wide_k1000_d50"]
    deep = picks["C_deep_k1000_d150"]
    cases = sorted(set.intersection(*(set(v) for v in picks.values())))

    out: dict[str, Any] = {
        "_comment": (
            "NR-47 / item 10 stage 2: two same-day arms over 37 cases, everything fixed but "
            "hyde.top_k (100 against 1000), sharing ONE pinned hypothesis set because NR-46 "
            "measured a redraw at +0.058 reach. Derived by evals/freeze_hyde_cut_arm.py; "
            "pinned by tests/test_hyde_cut_arm.py. The pre-registered kill condition -- net@2 "
            "must not fall -- FIRES. The papers the wider cut adds are fine (precision 0.882 "
            "against 0.878 kept); the digest SHRANK from a pool 5.9x larger, which is NR-11's "
            "dilution mechanism observed rather than inferred."
        ),
        # top_k is READ from each pool's recorded `pool_config`, never inferred from the arm's
        # name. The first version wrote `100 if "100" in name else 1000` and labelled the
        # treatment arm 100, because "treat_top_k_1000" contains "100" -- a substring test
        # standing in for a fact the artifact already holds. The pool records what it was
        # collected with; asking it is both shorter and correct.
        "arms": {
            name: {
                "run_file": f,
                "hyde_top_k": recorded_top_k(name),
                "gate_depth": GATE_DEPTH[name],
                "hypotheses_file": recorded_hyp_file(name),
                "pinned_hypotheses": bool(recorded_hyp_file(name)),
                "mean_net2": round(st.mean(net(picks[name][c]) for c in cases), 2),
                "digest_per_case": round(sum(len(picks[name][c]) for c in cases) / len(cases), 1),
                "precision": round(
                    sum(1 for c in cases for _p, sc in picks[name][c] if sc >= 2)
                    / sum(len(picks[name][c]) for c in cases),
                    3,
                ),
            }
            for name, f in ARMS.items()
        },
        "repaired_case": {
            "case": REPAIRED_CASE,
            "run_file": REPAIR,
            "why": (
                "Its control collection hit arXiv HTTP 429 after 10 attempts and 930s of "
                "throttle-waiting and fell back to a keyword-only pool with ZERO HyDE "
                "candidates. Paired against a treatment arm that had HyDE, its delta would "
                "have measured HyDE existing at all rather than the cut. Re-collected at "
                "identical flags once the throttle cleared, and spliced in rather than dropped."
            ),
        },
        "n_cases": len(cases),
        "cohorts": {},
        "per_case": {
            c: {
                "control": net(ctrl[c]),
                "treatment": net(treat[c]),
                "delta": net(treat[c]) - net(ctrl[c]),
            }
            for c in cases
        },
    }

    for label, pred in COHORTS.items():
        sel = [c for c in cases if pred(c)]
        d = [float(net(treat[c]) - net(ctrl[c])) for c in sel]
        lo, hi = paired_bootstrap(d)
        stt = sign_test(d)
        out["cohorts"][label] = {
            "n_cases": len(sel),
            "control_mean": round(st.mean(net(ctrl[c]) for c in sel), 2),
            "treatment_mean": round(st.mean(net(treat[c]) for c in sel), 2),
            "paired_delta": round(st.mean(d), 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "wins": stt["pos"],
            "losses": stt["neg"],
            "ties": stt["ties"],
            "sign_p": round(stt["p"], 4),
        }

    # -- the diagnostic: added-bad or displaced-good? --
    added, dropped, kept = [], [], []
    for c in cases:
        ci = dict(ctrl[c])
        ti = dict(treat[c])
        added += [(p, s) for p, s in ti.items() if p not in ci]
        dropped += [(p, s) for p, s in ci.items() if p not in ti]
        kept += [(p, s) for p, s in ti.items() if p in ci]
    n = len(cases)

    def prec(v):
        return round(sum(1 for _p, s in v if s >= 2) / len(v), 3) if v else None

    def value(v):
        return sum(pt(s) for _p, s in v)

    # Split the loss exactly into a COUNT effect and a RATE effect. With n_a added papers
    # worth v_a each and n_d dropped worth v_d, the delta is (n_a*v_a - n_d*v_d)/n, which
    # separates as (n_a - n_d)*v_d/n -- showing fewer papers at the rate we lost them --
    # plus n_a*(v_a - v_d)/n -- showing the ones we did at a different rate. The two sum to
    # the delta identically, so this is a decomposition and not a model of it.
    v_a = value(added) / len(added)
    v_d = value(dropped) / len(dropped)
    shrink = (len(added) - len(dropped)) * v_d / n
    quality = len(added) * (v_a - v_d) / n
    out["diagnostic"] = {
        "_comment": (
            "The fork stage 1 pre-registered. Papers the wider cut ADDS are as good as the "
            "ones already there, so the loss is not bad material -- the digest simply got "
            "smaller from a pool 5.9x larger, which is the gate reading a fixed depth of a "
            "diluted ranking. That is the 'the gate never saw them' branch."
        ),
        "kept": {"n": len(kept), "precision": prec(kept)},
        "added": {
            "n": len(added),
            "precision": prec(added),
            "net2_per_case": round(value(added) / n, 2),
        },
        "dropped": {
            "n": len(dropped),
            "precision": prec(dropped),
            "net2_forgone_per_case": round(-value(dropped) / n, 2),
        },
        "digest_size": {
            "control": round(sum(len(ctrl[c]) for c in cases) / n, 1),
            "treatment": round(sum(len(treat[c]) for c in cases) / n, 1),
        },
        "loss_split": {
            "_comment": (
                "Exact: the two terms sum to the paired delta. Most of the loss is the digest "
                "shrinking, not the papers being worse -- which is what makes gate_depth the "
                "follow-up rather than closing the direction outright."
            ),
            "from_showing_fewer": round(shrink, 3),
            "from_showing_worse": round(quality, 3),
            "sums_to_delta": round(shrink + quality, 3),
            "share_from_showing_fewer": round(shrink / (shrink + quality), 3),
            "value_per_added_paper": round(v_a, 3),
            "value_per_dropped_paper": round(v_d, 3),
        },
    }

    pc, ptr = pool_sizes("A_ships_k100_d50"), pool_sizes("B_wide_k1000_d50")
    common = sorted(set(pc) & set(ptr))
    out["pool_growth"] = {
        "control_per_case": round(sum(pc[c] for c in common) / len(common)),
        "treatment_per_case": round(sum(ptr[c] for c in common) / len(common)),
        "factor": round(sum(ptr[c] for c in common) / sum(pc[c] for c in common), 1),
    }

    # -- NR-48 / item 13: did widening the gate's window recover what the cut lost? --
    def paired(lo: str, hi: str, sel) -> dict[str, Any]:
        d = [float(net(picks[hi][c]) - net(picks[lo][c])) for c in sel]
        lo_ci, hi_ci = paired_bootstrap(d)
        stt = sign_test(d)
        return {
            "paired_delta": round(st.mean(d), 2),
            "ci95": [round(lo_ci, 2), round(hi_ci, 2)],
            "wins": stt["pos"],
            "losses": stt["neg"],
            "ties": stt["ties"],
            "sign_p": round(stt["p"], 4),
        }

    d_added, d_dropped = [], []
    for c in cases:
        bi, ci2 = dict(treat[c]), dict(deep[c])
        d_added += [(p, s) for p, s in ci2.items() if p not in bi]
        d_dropped += [(p, s) for p, s in bi.items() if p not in ci2]
    out["depth_arm"] = {
        "_comment": (
            "Item 13: hold hyde.top_k at 1000 and move gate_depth 50 -> 150, on the pools B "
            "already collected. Pre-registered: the digest must return to at least the "
            "shipped 8.3/case AND net@2 to at least +5.51. The digest recovered; net@2 did "
            "not. That is the pre-registered kill -- the papers a deeper window admits are "
            "worse than the ones they displace -- and it closes the direction."
        ),
        "vs_ships": paired("A_ships_k100_d50", "C_deep_k1000_d150", cases),
        "vs_wide": paired("B_wide_k1000_d50", "C_deep_k1000_d150", cases),
        "cohorts_vs_ships": {
            label: paired("A_ships_k100_d50", "C_deep_k1000_d150", [c for c in cases if pred(c)])
            for label, pred in COHORTS.items()
        },
        "what_depth_changed": {
            "added": {"n": len(d_added), "precision": prec(d_added)},
            "dropped": {"n": len(d_dropped), "precision": prec(d_dropped)},
        },
        "pre_registered": {
            "digest_bar_per_case": 8.3,
            "net2_bar": 5.51,
            "digest_recovered": bool(
                round(sum(len(deep[c]) for c in cases) / len(cases), 1) >= 8.3
            ),
            "net2_recovered": bool(st.mean(net(deep[c]) for c in cases) >= 5.51),
        },
    }

    all37 = out["cohorts"]["all37"]
    out["verdict"] = {
        "pre_registered_kill": "net@2 must not fall",
        "net2_fell": bool(all37["paired_delta"] < 0),
        "killed": bool(all37["paired_delta"] < 0),
        "ci_excludes_zero": bool(all37["ci95"][1] < 0),
        "sign_test_resolves": bool(all37["sign_p"] < 0.05),
        "added_papers_are_good": bool(
            out["diagnostic"]["added"]["precision"] >= out["diagnostic"]["kept"]["precision"]
        ),
        "digest_shrank": bool(
            out["diagnostic"]["digest_size"]["treatment"]
            < out["diagnostic"]["digest_size"]["control"]
        ),
        "follow_up_tested": "gate_depth 50 -> 150 [NR-48, item 13]",
        "depth_recovered_the_digest": out["depth_arm"]["pre_registered"]["digest_recovered"],
        "depth_recovered_net2": out["depth_arm"]["pre_registered"]["net2_recovered"],
        "direction_closed": not out["depth_arm"]["pre_registered"]["net2_recovered"],
        "closing_note": (
            "Both levers are spent. Widening retrieval shrank the digest; widening the gate's "
            "window regrew it and net@2 still did not recover -- deeper admissions run at 0.859 "
            "precision against the 0.951 they displace. The wide+deep combination is a WASH "
            "against what ships (-0.19, CI spanning zero, 14w/15l) at 5.9x the pool and 3x the "
            "gate calls: no gain, real cost."
        ),
    }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: out[k] for k in ("cohorts", "diagnostic", "pool_growth", "verdict")}, indent=1
        )
    )
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
