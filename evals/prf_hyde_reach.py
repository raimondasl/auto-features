"""Item 12 stage 1: does a second, feedback-seeded HyDE round aim better? [NR-49]

The only open evidence-led lead, and NR-48 changed what makes it interesting. Widening
retrieval (NR-47) and widening the gate's window (NR-48) both failed, so **more candidates is a
closed direction**. Pseudo-relevance feedback earns a look only if it aims *better* at the same
budget -- not if it simply fetches more.

**So the comparison is budget-matched, and that is the whole design.**

    baseline   round 1 at cut 100                      ~400 candidate slots
    treatment  round 1 at cut 50  UNION  round 2 at 50 ~400 candidate slots

Testing round-1@100 against round-1@100 + round-2@100 would re-run NR-47's experiment with
extra steps: it would almost certainly raise reach, and NR-48 already established that raising
reach this way does not raise net@2. The honest question is whether a second round *re-aimed by
what the gate admitted* beats simply looking deeper with the first aim.

**PRE-REGISTERED, written before the round-2 hypotheses existed.**

* Baseline is NR-46's measured reach for the pinned round-1 set at cut 100: **0.2231**.
* **Bar: the budget-matched union must beat 0.2231.** Anything at or below it means a second
  round is worth less than looking twice as deep with the first, and item 12 dies here for the
  price of 37 LLM calls.
* The unequal-budget number (both rounds at 100) is computed and reported, and is **not** the
  decision criterion. It is there so nobody has to wonder what was hidden.

**How round 2 is seeded.** From the papers the shipped arm actually *showed* -- gate-admitted,
finescale-passed, window-cut -- because that is the strongest statement the system makes about
what it found useful. Their abstracts are appended **after** `repo_context_block`, never merged
into it: the fine-scale stage's probability map is fitted to those exact bytes, and P-era work
already measured what happens when stated wants are folded into the gate's own question
(net@2 +57 against +95, the worst arm in the campaign). Feedback belongs in the query.

**And reach is necessary, not sufficient.** Item 10's stage 1 passed at +101% and its paid arm
still lost 0.78 net@2. A win here buys a *cheaper* paid arm, not a likely one.

    uv run python evals/prf_hyde_reach.py --hypotheses   # 37 LLM calls, one-time
    uv run python evals/prf_hyde_reach.py                # $0, ~12 min of CPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id  # noqa: E402

WORK = EVALS / ".work"
INDEX = WORK / "hyde_index"
POOL = "pool-cohort3"
SEED_POOL = WORK / "pool-cut100"  # where the shipped arm's candidate abstracts live
ROUND1_HYP = WORK / "hyde_hypotheses_pinned.json"
ROUND2_HYP = WORK / "hyde_hypotheses_prf.json"
ROUND1_RANKS = WORK / "hyde_witness_ranks.json"  # NR-46's, same pinned set
ROUND2_RANKS = WORK / "prf_witness_ranks.json"
SHIPPED_RUN = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T034455Z.json"
REPAIR_RUN = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T075622Z.json"
FROZEN = EVALS / "prf_hyde_reach.json"

BASELINE_CUT = 100  # round 1 alone, the budget-matched baseline
SPLIT_CUT = 50  # each round in the matched union
HYP_MODEL = "claude-haiku-4-5"  # round 1 model; see generate()
BAR = 0.2231  # NR-46's measured round-1 reach at cut 100

PRF_BLOCK = """

Papers already surfaced for this repository, which its own pipeline judged worth showing.
Treat them as evidence of the direction that works, and write abstracts for the papers they
point TOWARDS but do not themselves cover -- adjacent problems, the next step, the technique
these imply. Do not restate them.

{seeds}
"""


def shown_by_case() -> dict[str, list[str]]:
    """What the shipped arm showed, per case — the feedback signal."""
    res = EVALS / "results"
    run = json.loads((res / SHIPPED_RUN).read_text(encoding="utf-8"))
    out = {
        e["case"]: [dedup_id(str(p["arxiv_id"])) for p in e["returned"]["reporadar_toppicks"]]
        for e in run
    }
    repair = json.loads((res / REPAIR_RUN).read_text(encoding="utf-8"))
    for e in repair:  # bio-mdtraj's control degraded; its repair is the real shipped answer
        out[e["case"]] = [dedup_id(str(p["arxiv_id"])) for p in e["returned"]["reporadar_toppicks"]]
    return out


def abstracts_for(case: str, ids: list[str], limit: int = 6) -> list[str]:
    f = SEED_POOL / f"{case}.json"
    if not f.is_file():
        return []
    by_id = {
        dedup_id(str(c["arxiv_id"])): (c.get("abstract") or "").strip()
        for c in json.loads(f.read_text(encoding="utf-8"))["candidates"]
    }
    got = [by_id[i][:700] for i in ids if by_id.get(i)]
    return got[:limit]


def generate() -> int:
    import os
    from types import SimpleNamespace

    from harness import profile_case_repo
    from run_judge_eval import load_dotenv

    from reporadar import hyde
    from reporadar.triage import repo_context_block

    load_dotenv(EVALS / ".env")
    shown = shown_by_case()
    have = json.loads(ROUND2_HYP.read_text(encoding="utf-8")) if ROUND2_HYP.is_file() else {}
    # Same model and same call path round 1 used (`--rr-triage-model` default). A round 2
    # written by a different model would measure the model, not the feedback.
    cfg = SimpleNamespace(
        provider="claude",
        claude_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        claude_model=HYP_MODEL,
        timeout=120,
    )
    todo = [c for c in sorted(shown) if c not in have and (WORK / c).is_dir()]
    print(f"{len(todo)} case(s) need round-2 hypotheses")
    for n, case in enumerate(todo, start=1):
        seeds = abstracts_for(case, shown[case])
        if not seeds:
            # A case the shipped arm showed nothing for has no feedback to give. Recorded and
            # skipped -- inventing a seed would make round 2 a fresh draw wearing PRF's name.
            print(f"  [{n}/{len(todo)}] {case:<16} no shown papers — skipped, not imputed")
            continue
        profile = profile_case_repo(WORK / case)
        context = repo_context_block(profile)[:6000] + PRF_BLOCK.format(
            seeds="\n\n".join(f"- {a}" for a in seeds)
        )
        prompt = hyde.HYPOTHESIS_PROMPT.format(n=4, context=context)
        try:
            from reporadar.llm_client import complete

            raw = complete(prompt, cfg, max_tokens=2500)
            start, end = raw.find("["), raw.rfind("]")
            items = [str(x).strip() for x in json.loads(raw[start : end + 1]) if str(x).strip()]
        except Exception as exc:  # noqa: BLE001 — one bad case must not lose the others
            print(f"  [{n}/{len(todo)}] {case:<16} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        have[case] = items[:4]
        ROUND2_HYP.write_text(json.dumps(have, indent=1), encoding="utf-8")
        print(f"  [{n}/{len(todo)}] {case:<16} {len(have[case])} from {len(seeds)} seed(s)")
    return 0


def judged_scores() -> dict[tuple[str, str], int]:
    """Every (case, paper) any experiment ever judged.

    **C-35's caveat is load-bearing and this call site respects it.** This union is not a
    sample of product output — `diagnose_ranker.py` judged ranks 151+, so membership says
    nothing about what RepoRadar shows. It is used here only to look up a score for a paper
    named on other grounds, which is the one question it can answer. Papers absent from it
    are **void, not null**: they were never scored, which is not the same as scoring badly.
    """
    out: dict[tuple[str, str], int] = {}
    for f in sorted((EVALS / "results").glob("judge-*.json")):
        try:
            run = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(run, list):
            continue
        for e in run:
            for grp in (e.get("returned") or {}).values():
                if not isinstance(grp, list):
                    continue
                for p in grp:
                    if isinstance(p, dict) and "arxiv_id" in p and "judge_score" in p:
                        out[(e.get("case"), dedup_id(str(p["arxiv_id"])))] = int(p["judge_score"])
    return out


def witnesses_by_case() -> dict[str, list[tuple[str, list[str]]]]:
    import witness_set as ws

    data = json.loads((EVALS / "witness_set.json").read_text(encoding="utf-8"))["witnesses"]
    out = {}
    for case, papers in data.items():
        rows = [
            (pid, sorted(ws.non_self_sources(m["sources"])))
            for pid, m in papers.items()
            if ws.non_self_sources(m["sources"])
        ]
        if rows:
            out[case] = rows
    return out


def compute_ranks(cases) -> dict[str, dict[str, int]]:
    import numpy as np

    from reporadar import hyde

    hyp = json.loads(ROUND2_HYP.read_text(encoding="utf-8"))
    shards = hyde.index_shards(INDEX)
    ids_all: list[str] = []
    for s in shards:
        ids_all.extend(
            dedup_id(i) for i in (INDEX / f"{s.stem}.ids").read_text(encoding="utf-8").split("\n")
        )
    pos = {pid: i for i, pid in enumerate(ids_all)}
    model = hyde.load_encoder()
    ok, dists = hyde.verify_encoder(model)
    if not ok:
        raise SystemExit(f"encoder does not reproduce the index (Hamming {dists}); refusing")

    out: dict[str, dict[str, int]] = {}
    todo = sorted(c for c in cases if c in hyp)
    for n, case in enumerate(todo, start=1):
        want = [pid for pid, _s in cases[case] if pid in pos]
        if not want:
            continue
        bits = hyde.encode_binary(model, list(hyp[case]))
        best = dict.fromkeys(want, 10**9)
        for row in range(bits.shape[0]):
            d = np.concatenate(
                [hyde._hamming(np.load(s, mmap_mode="r"), bits[row]) for s in shards]
            )
            for pid in want:
                best[pid] = min(best[pid], int((d < d[pos[pid]]).sum()))
        out[case] = best
        print(f"  [{n}/{len(todo)}] {case:<16} {len(want):>3} witnesses ranked")
    ROUND2_RANKS.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hypotheses", action="store_true")
    ap.add_argument("--ranks", action="store_true")
    args = ap.parse_args()
    if args.hypotheses:
        return generate()

    import witness_set as ws

    cases = witnesses_by_case()
    pools = ws._pool_ids(POOL)
    r1 = json.loads(ROUND1_RANKS.read_text(encoding="utf-8"))
    r2 = (
        compute_ranks(cases)
        if args.ranks or not ROUND2_RANKS.is_file()
        else json.loads(ROUND2_RANKS.read_text(encoding="utf-8"))
    )
    hyp2 = json.loads(ROUND2_HYP.read_text(encoding="utf-8"))

    rows = []
    for case, items in cases.items():
        if case not in pools:
            continue
        for pid, sources in items:
            rows.append(
                (
                    case,
                    pid,
                    sources,
                    pid in pools[case],
                    r1.get(case, {}).get(pid),
                    r2.get(case, {}).get(pid),
                )
            )

    def reach(pred) -> dict[str, Any]:
        hit = [r for r in rows if r[3] or pred(r[4], r[5])]
        return {"n": len(rows), "reached": len(hit), "p": round(len(hit) / len(rows), 4)}

    def under(r: int | None, k: int) -> bool:
        return r is not None and r < k

    baseline = reach(lambda a, _b: under(a, BASELINE_CUT))
    matched = reach(lambda a, b: under(a, SPLIT_CUT) or under(b, SPLIT_CUT))
    r1_only_50 = reach(lambda a, _b: under(a, SPLIT_CUT))
    unequal = reach(lambda a, b: under(a, BASELINE_CUT) or under(b, BASELINE_CUT))
    # The unequal arm spends ~800 slots. So does round 1 at 200 -- which makes THAT its budget
    # match, and the honest test of whether its larger number is round 2 or merely depth.
    r1_only_200 = reach(lambda a, _b: under(a, 2 * BASELINE_CUT))

    # ── Is +0.0057 a result? The reach table alone cannot say, so these decide it. ──────
    fresh = [r for r in rows if not r[3]]  # not already in the frozen pool
    in_m = [r for r in fresh if under(r[4], SPLIT_CUT) or under(r[5], SPLIT_CUT)]
    in_b = [r for r in fresh if under(r[4], BASELINE_CUT)]
    gained = [r for r in in_m if r not in in_b]
    lost = [r for r in in_b if r not in in_m]
    mcnemar = binomtest(len(gained), len(gained) + len(lost), 0.5).pvalue

    # Witnesses round 1 buries *past NR-47's widest measured cut* that round 2 surfaces.
    # Width cannot buy these at any cut this project has run.
    deep = [r for r in fresh if r[4] is not None and r[4] > 1000 and under(r[5], BASELINE_CUT)]
    scores = judged_scores()
    unique_judged = [scores[(r[0], r[1])] for r in gained if (r[0], r[1]) in scores]

    out: dict[str, Any] = {
        "_comment": (
            "NR-49 / item 12 stage 1: does a feedback-seeded second HyDE round aim better at "
            "the SAME candidate budget? Derived by evals/prf_hyde_reach.py; pinned by "
            "tests/test_prf_hyde_reach.py. The decision criterion is the BUDGET-MATCHED union "
            "(round 1 at 50 union round 2 at 50) against round 1 alone at 100. The "
            "unequal-budget figure is reported and is NOT the criterion: NR-47 and NR-48 "
            "already established that buying reach with more candidates does not raise net@2."
        ),
        "pre_registered": {
            "bar": BAR,
            "bar_is": "NR-46's measured round-1 reach at cut 100, same pinned hypotheses",
            "criterion": "budget_matched union must beat the bar",
            "written_before_round2_existed": True,
        },
        "prf_blind_spot": {
            "_comment": (
                "PRF needs something to feed on. Four cases produced no round 2 because the "
                "shipped arm SHOWED NOTHING there -- the abstention that makes RepoRadar "
                "competitive with Opus 5 is also what starves feedback. They are skipped, "
                "never imputed: a fresh draw for them would be a round-1 redraw wearing PRF's "
                "name, and NR-46 measured that redraw at +0.0577 reach, ten times the effect "
                "under test. So the method is structurally blind to these witnesses, and they "
                "are the hardest ones -- none is in the pool by any route."
            ),
            "cases": sorted(set(cases) - set(hyp2)),
            "witnesses_in_them": sum(1 for r in rows if r[0] not in hyp2),
            "of_those_already_reached": sum(1 for r in rows if r[0] not in hyp2 and r[3]),
        },
        "cases_with_round2": len(hyp2),
        "witnesses": len(rows),
        "reach": {
            "round1_at_50": r1_only_50,
            "round1_at_100_BASELINE": baseline,
            "budget_matched_50_plus_50": matched,
            "round1_at_200": r1_only_200,
            "unequal_100_plus_100_not_the_criterion": unequal,
        },
    }
    out["discordant"] = {
        "_comment": (
            "The reach delta is a net of two flows and the net is what the bar reads. These "
            "are the flows. Losses are mechanical -- every one is a round-1 rank in [50,100) "
            "dropped by halving the cut, nothing to do with round 2's aim."
        ),
        "gained": len(gained),
        "lost": len(lost),
        "net": len(gained) - len(lost),
        "mcnemar_exact_p": round(float(mcnemar), 3),
        "resolves": bool(mcnemar < 0.05),
        "lost_round1_ranks": sorted(int(r[4]) for r in lost),
        "gained_round1_rank_median": (
            sorted(int(r[4]) for r in gained if r[4] is not None)[len(gained) // 2]
            if gained
            else None
        ),
    }
    g2 = [r for r in fresh if (under(r[4], 100) or under(r[5], 100)) and not under(r[4], 200)]
    l2 = [r for r in fresh if under(r[4], 200) and not (under(r[4], 100) or under(r[5], 100))]
    out["second_budget_point"] = {
        "_comment": (
            "The unequal arm's 0.2712 looked like the one encouraging number in the table. It "
            "is not: at its OWN budget match -- round 1 alone at 200, same ~800 slots -- round "
            "1 reaches 0.2692. The whole apparent gain was depth. Two independent budget "
            "points now say the same thing, and this one says it at p = 1.00."
        ),
        "slots": 800,
        "round1_at_200": r1_only_200["p"],
        "round1_100_union_round2_100": unequal["p"],
        "delta": round(unequal["p"] - r1_only_200["p"], 4),
        "gained": len(g2),
        "lost": len(l2),
        "mcnemar_exact_p": round(float(binomtest(len(g2), len(g2) + len(l2), 0.5).pvalue), 4),
    }
    out["reaches_what_width_cannot"] = {
        "_comment": (
            "POST HOC. Not what the bar asked, generated by looking at the result, and it "
            "does NOT rescue the null above -- it is recorded as a hypothesis needing its own "
            "pre-registered test. NR-47's widest measured cut was 1000; these sit past it, so "
            "no amount of widening round 1 reaches them. That is a different mechanism from "
            "the one NR-47 and NR-48 spent, which added ranks 100-1000 of the SAME query."
        ),
        "n": len(deep),
        "examples": [
            {"case": r[0], "paper": r[1], "round1": int(r[4]), "round2": int(r[5])}
            for r in sorted(deep, key=lambda x: x[5])[:6]
        ],
        "prf_unique_ever_judged": {
            "judged": len(unique_judged),
            "of": len(gained),
            "actionable": sum(1 for s in unique_judged if s >= 2),
            "never_scored_void_not_null": len(gained) - len(unique_judged),
        },
    }
    out["verdict"] = {
        "budget_matched_reach": matched["p"],
        "baseline_reach": baseline["p"],
        "delta": round(matched["p"] - baseline["p"], 4),
        "clears_the_bar_as_written": bool(matched["p"] > BAR),
        "redraw_noise_floor": 0.0577,
        "delta_vs_noise_floor": round((matched["p"] - baseline["p"]) / 0.0577, 2),
        "result_is_null": True,
        "null_at_two_independent_budget_points": {
            "400_slots": {"delta": 0.0057, "mcnemar_p": 0.678},
            "800_slots": {"delta": 0.0019, "mcnemar_p": 1.0},
        },
        "the_bar_was_underspecified": (
            "It named a threshold and no minimum effect size, so a null cleared it by three "
            "witnesses of 520. NR-46 measured a plain hypothesis REDRAW -- same cut, same "
            "method, different draw -- at +0.0577 reach. This is +0.0057, a tenth of the "
            "noise floor of the procedure it modifies, at McNemar p = 0.68. The pass is "
            "reported and refused: a stage-1 gate exists to license spending, and this does "
            "not. Recorded as a defect in the pre-registration, not repaired after the fact."
        ),
        "licenses_paid_arm": False,
        "reach_is_necessary_not_sufficient": (
            "Item 10's stage 1 passed at +101% and its paid arm lost 0.78 net@2 (NR-47). Even "
            "a real reach win would have bought a cheaper paid arm, not a likely one."
        ),
    }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("reach", "verdict")}, indent=1))
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
