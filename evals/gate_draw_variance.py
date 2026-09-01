"""How much of RepoRadar's run-to-run variation is the GATE, not the pool? [NR-54]

NR-52 noticed the shipped arm's mean net@2 moves between two of our own runs (+5.73 on
2026-08-27, +5.51 on 2026-08-30) and filed it as draw noise under C-7. Following that up
produced a per-case sd of **2.23** between those runs — and then a check that **corrects the
framing**: the two runs used *different pool directories*, `pool-core25-arxiv` and
`pool-cut100`, whose candidate sets share a **median Jaccard of only 0.365**. That 2.23
confounds pool-collection drift with everything downstream and cannot separate them. Any claim
built on it — including the suggestion that the benchmark's resolution is gate-limited — was
premature.

**So this holds the pool byte-identical and re-runs.** Same frozen pool (`pool-cut100`), same
flags, same fingerprint, so retrieval contributes exactly nothing and what remains is:

* the **actionability gate** — Haiku, uncached, and `_call_claude` sends **no temperature**, so
  the Anthropic default of 1.0 applies and every verdict is a sample (NR-53 measured the same
  omission on the judge path at an 8.4% label-flip rate);
* the **fine-scale rescore** — OpenAI logprobs, expected deterministic, and this run tests that
  expectation rather than assuming it;
* the **judge** — cached per `(model, repo, paper)`, so papers already scored return identically
  and only newly-shown papers cost anything.

**PRE-REGISTERED, written before the replicate run.** This is descriptive — there is no
treatment and nothing to kill — so what is fixed in advance is the *reading*, which is where a
result like this can otherwise be spun after the fact:

* **gate sd < 1.0 per case** — downstream stochasticity is minor. The 2.23 is mostly pool
  collection, the benchmark's ±0.78 resolution is not gate-limited, and setting temperature 0
  is reproducibility hygiene with no measurement dividend. **Frozen pools already bought the
  sensitivity that matters.**
* **gate sd >= 1.5 per case** — downstream stochasticity dominates the paired noise in every
  frozen-pool arm this project has run, including NR-47/48/51. Temperature 0 would then sharpen
  every future arm, and the ladder in `RESEARCH-net2-directions.md` becomes measurable at
  effects it currently cannot resolve.
* between: report both readings and decide in the open.

**The decomposition, fixed now.** Treating the sources as independent,
`sd_total^2 = sd_pool^2 + sd_gate^2` with `sd_total = 2.23` measured. Whatever `sd_gate` comes
back as, the pool share follows as `sqrt(2.23^2 - sd_gate^2)` — and that arithmetic is stated
here so it cannot be rearranged later to favour whichever component looks more interesting.

    uv run python evals/gate_draw_variance.py --plan   # $0: what the replicate will cost
    uv run python evals/gate_draw_variance.py          # $0: the decomposition, after the run
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id  # noqa: E402

RES = EVALS / "results"
BASE = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T034455Z.json"
BASE_REPAIR = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T075622Z.json"
REPLICATE = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260901T040926Z.json"
FROZEN = EVALS / "gate_draw_variance.json"

SD_TOTAL = 2.23  # measured between the two DIFFERENT-pool runs; the thing being decomposed
POOL_JACCARD = 0.365  # median candidate overlap between those two pools
LOW, HIGH = 1.0, 1.5  # pre-registered reading thresholds


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


def plan() -> int:
    base = load(BASE)
    n_papers = sum(len(v) for v in base.values())
    print("replicate command (reuses pool-cut100 -- retrieval contributes nothing):\n")
    print(
        "  uv run python evals/run_judge_eval.py --baseline none --sources arxiv --rr-all-time \\\n"
        "    --rr-bigrams verified --rr-hyde --rr-hyde-index evals/.work/hyde_index \\\n"
        "    --rr-hyde-hypotheses-file evals/.work/hyde_hypotheses_pinned.json \\\n"
        "    --rr-hyde-top-k 100 --rr-triage --rr-rerank --rr-hybrid --rr-finescale \\\n"
        "    --rr-pool 50 --rr-window 15 --rr-w-embedding 1.5 \\\n"
        "    --rr-frozen-pool evals/.work/pool-cut100"
    )
    print(f"\ngate calls: 50/case x {len(base)} cases = {50 * len(base)} Haiku verdicts")
    print(f"baseline digest: {n_papers} papers, judge verdicts cached per (model, repo, paper)")
    print("only NEWLY shown papers incur fresh judging; the rest return from cache")
    return 0


def report() -> int:
    if not REPLICATE:
        raise SystemExit("set REPLICATE to the replicate run's filename first (see --plan)")
    base = load(BASE)
    base["bio-mdtraj"] = load(BASE_REPAIR)["bio-mdtraj"]
    rep = load(REPLICATE)
    cases = sorted(set(base) & set(rep))

    d = [float(net(rep[c]) - net(base[c])) for c in cases]
    sd_gate = st.stdev(d)
    identical = sum(1 for c in cases if base[c] == rep[c])
    jac = [
        len({p for p, _ in base[c]} & {p for p, _ in rep[c]})
        / max(1, len({p for p, _ in base[c]} | {p for p, _ in rep[c]}))
        for c in cases
    ]
    pool_share = (SD_TOTAL**2 - sd_gate**2) ** 0.5 if sd_gate < SD_TOTAL else 0.0

    out: dict[str, Any] = {
        "_comment": (
            "NR-54. Holds the pool BYTE-IDENTICAL (same frozen pool, same flags, same "
            "fingerprint) and re-runs the shipped config, so retrieval contributes nothing and "
            "what varies is the gate (Haiku, uncached, no temperature sent -> default 1.0), the "
            "fine-scale rescore, and newly-judged papers. Corrects NR-52's follow-up framing: "
            "the 2.23 sd quoted there was measured between runs on DIFFERENT pools sharing a "
            "median Jaccard of 0.365, so it could never have isolated the gate. Derived by "
            "evals/gate_draw_variance.py; pinned by tests/test_gate_draw_variance.py."
        ),
        "pre_registered": {
            "descriptive_not_a_kill": True,
            "reading_below": {
                "sd_under": LOW,
                "means": (
                    "gate stochasticity minor; the 2.23 is mostly pool collection; "
                    "temperature 0 is hygiene with no measurement dividend"
                ),
            },
            "reading_above": {
                "sd_at_least": HIGH,
                "means": (
                    "gate stochasticity dominates paired noise in every frozen-pool arm; "
                    "temperature 0 would sharpen future arms"
                ),
            },
            "decomposition": "sd_total^2 = sd_pool^2 + sd_gate^2, sd_total = 2.23 measured",
            "written_before_the_replicate": True,
        },
        "pool_held_identical": True,
        "n_cases": len(cases),
        "gate_sd_per_case": round(sd_gate, 2),
        "mean_delta": round(st.mean(d), 2),
        "cases_byte_identical": identical,
        "median_digest_jaccard": round(st.median(jac), 3),
        "context": {
            "sd_total_across_different_pools": SD_TOTAL,
            "pool_median_jaccard": POOL_JACCARD,
            "implied_pool_component_sd": round(pool_share, 2),
        },
    }
    # What the grey band asked for: the actual measurement dividend, computed in the open.
    # NR-47's paired arm reported ci95 [-1.59, -0.03] around -0.78, so its half-width is 0.78
    # and its paired sd is 0.78 * sqrt(37) / 1.96.
    import math

    sd_arm = 0.78 * math.sqrt(len(cases)) / 1.96
    sd_treat = math.sqrt(max(0.0, sd_arm**2 - sd_gate**2))
    hw_new = 1.96 * sd_treat / math.sqrt(len(cases))
    out["measurement_dividend"] = {
        "_comment": (
            "The number the decision actually turns on, and it is smaller than the sd alone "
            "suggests: resolution scales with the square root of variance, so removing 35% of "
            "the variance tightens the interval by 20%, not by 35%. An earlier claim in "
            "conversation -- that the resolution might go from +-0.78 to ~+-0.30 -- assumed the "
            "whole 2.23 was gate noise. It is 1.44, and this is the corrected figure."
        ),
        "nr47_observed_paired_sd": round(sd_arm, 2),
        "gate_share_of_variance": round((sd_gate / sd_arm) ** 2, 3),
        "residual_treatment_sd": round(sd_treat, 2),
        "half_width_now": 0.78,
        "half_width_if_gate_deterministic": round(hw_new, 2),
        "tighter_by": round(1 - hw_new / 0.78, 3),
        "ladder_rungs_still_unresolvable": bool(hw_new > 0.45),
    }

    out["verdict"] = {
        "gate_dominates": bool(sd_gate >= HIGH),
        "gate_minor": bool(sd_gate < LOW),
        "grey": bool(LOW <= sd_gate < HIGH),
        "temperature_fix_has_measurement_dividend": bool(sd_gate >= HIGH),
        "reading": (
            "GREY, and the grey resolves toward 'worth doing, not transformative'. The gate is "
            "35% of the paired variance in a frozen-pool arm, so making it deterministic "
            "tightens the resolution from +-0.78 to ~+-0.63 -- 20%. Real, cheap, and NOT enough "
            "to rescue the ladder: its rungs run +0.20 to +0.45 and remain individually "
            "unresolvable, so the bundle-only rule stands."
        ),
        "pool_is_the_larger_component": bool(pool_share > sd_gate),
    }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"{len(cases)} cases, pool held byte-identical\n")
    print(f"  per-case net@2 delta : mean {st.mean(d):+.2f}, sd {sd_gate:.2f}")
    print(f"  cases byte-identical : {identical}/{len(cases)}")
    print(f"  median digest Jaccard: {st.median(jac):.3f}")
    print(f"\n  total sd across different pools : {SD_TOTAL}")
    print(f"  gate/downstream component       : {sd_gate:.2f}")
    print(f"  implied pool component          : {pool_share:.2f}")
    band = "GATE DOMINATES" if sd_gate >= HIGH else ("GATE MINOR" if sd_gate < LOW else "GREY")
    print(f"\nPRE-REGISTERED READING: sd < {LOW} minor, >= {HIGH} dominates  ->  {band}")
    print(f"wrote {FROZEN.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()
    return plan() if args.plan else report()


if __name__ == "__main__":
    raise SystemExit(main())
