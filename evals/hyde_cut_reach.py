"""Item 10, stage 1: would a wider HyDE union actually reach more witnesses? [NR-46]

NR-45 measured 164 of Opus 5's 302 actionable picks sitting in our own dense index and never
reaching our pool, at **median rank 1,087 against a cut of 100**. That says the union is too
narrow. It does not say a wider one helps, and this project has a scar on exactly that point:
**NR-11** recorded a wider pool meeting a near-binary gate and making the headline *worse*.

So stage 1 asks the cheap question first -- does reach move at all -- and it asks it without
re-collecting a single pool. A witness is reached at cut *K* if it is **already in the frozen
pool, or its HyDE rank is below K**. Both halves are computable from artifacts on disk: pool
membership from `pool-cohort3`, ranks from the shipped index under the shipped distance
function. No arXiv calls, no judge calls, and a Tier B run only if this passes.

**PRE-REGISTERED, written before the measurement was run.**

Pooled non-self reach into `pool-cohort3` is **0.165** (86 of 520 witnesses). NR-45's rank
curve recovers 49% of missing papers at cut 1,000. If witnesses behave like Opus 5's picks:

* **Bar: pooled non-self reach >= 0.25 at K = 1000** -- a rise of >= 0.085, better than 50%
  relative. Clears it, and a Tier B arm is worth its $25.
* **Kill: < 0.20.** If widening the cut tenfold cannot lift reach past 0.20, the premise is
  wrong and item 10 dies here, having cost nothing but CPU.
* Between the two: marginal, reported as such and decided rather than rounded up.

**What the simulation cannot see, stated up front.** It adds papers HyDE would have returned
at a wider cut; it cannot model second-order effects of a bigger pool -- the ranker's
normalisation shifts, and `gate_depth` still shows the gate only its top 50. Those are exactly
where NR-11's damage lived, and they need the paid run. This bounds reach from below and
decides only whether that run is worth buying.

**Hypotheses for 17 of the 37 cases did not exist** -- including all six materials cases, which
NR-45 identified as both the cohort we lose outright and the one where 61.8% of losses are this
stage. They hold 184 of the 520 non-self witnesses, so measuring without them would have
answered for two thirds of the evidence. `--hypotheses` generates them (one LLM call each,
~$0.20 total) into a separate file, leaving the replication's frozen cache untouched.

    uv run python evals/hyde_cut_reach.py --hypotheses   # ~17 LLM calls, one-time
    uv run python evals/hyde_cut_reach.py                # $0, ~12 min of CPU
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id, is_arxiv_id  # noqa: E402

WORK = EVALS / ".work"
INDEX = WORK / "hyde_index"
POOL = "pool-cohort3"
FROZEN_HYP = WORK / "hyde_hypotheses.json"  # the replication's; read-only here
EXTRA_HYP = WORK / "hyde_hypotheses_extra.json"  # ours, for the 17 it never covered
RANKS = WORK / "hyde_witness_ranks.json"
FROZEN = EVALS / "hyde_cut_reach.json"
WITNESSES = EVALS / "witness_set.json"

SHIPPED_CUT = 100
CUTS = (100, 200, 400, 1000, 2000, 5000, 10000)
BAR = 0.25  # pre-registered, at K = 1000
KILL = 0.20


def load_hypotheses() -> dict[str, list[str]]:
    """The replication's frozen cache plus ours, never merged into its file.

    Kept apart because that cache is what makes the P4 arms comparable to each other; adding
    to it would quietly change what "the cached hypotheses" means for every earlier result.
    """
    hyp = json.loads(FROZEN_HYP.read_text(encoding="utf-8"))
    if EXTRA_HYP.is_file():
        hyp = {**hyp, **json.loads(EXTRA_HYP.read_text(encoding="utf-8"))}
    return hyp


def witnesses_by_case() -> dict[str, list[tuple[str, list[str]]]]:
    """Non-self witnesses only: reporadar's own picks are in the pool by construction."""
    import witness_set as ws

    data = json.loads(WITNESSES.read_text(encoding="utf-8"))["witnesses"]
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for case, papers in data.items():
        rows = [
            (pid, sorted(ws.non_self_sources(m["sources"])))
            for pid, m in papers.items()
            if ws.non_self_sources(m["sources"])
        ]
        if rows:
            out[case] = rows
    return out


def generate() -> int:
    from dotenv import load_dotenv
    from harness import profile_case_repo

    from reporadar import hyde
    from reporadar.config import SuggestionsConfig

    load_dotenv(EVALS / ".env")
    have = load_hypotheses()
    need = [c for c in sorted(witnesses_by_case()) if c not in have]
    print(f"{len(need)} case(s) need hypotheses: {need}")
    if not need:
        return 0
    cfg = SuggestionsConfig(provider="claude", timeout=120)
    extra = json.loads(EXTRA_HYP.read_text(encoding="utf-8")) if EXTRA_HYP.is_file() else {}
    for n, case in enumerate(need, start=1):
        repo = WORK / case
        if not repo.is_dir():
            print(f"  [{n}/{len(need)}] {case:<16} NO CLONE — skipped, not imputed")
            continue
        try:
            hs = hyde.generate_hypotheses(profile_case_repo(repo), cfg, n=4)
        except Exception as exc:  # noqa: BLE001 — one bad case must not lose the others
            print(f"  [{n}/{len(need)}] {case:<16} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        extra[case] = hs
        EXTRA_HYP.write_text(json.dumps(extra, indent=1), encoding="utf-8")
        print(f"  [{n}/{len(need)}] {case:<16} {len(hs)} hypotheses  ({hs[0][:60]}...)")
    return 0


def compute_ranks(cases: dict[str, list[tuple[str, list[str]]]]) -> dict[str, dict[str, int]]:
    """Best rank of each witness across its case's hypotheses, over the shipped index."""
    import numpy as np

    from reporadar import hyde

    hyp = load_hypotheses()
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
    RANKS.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hypotheses", action="store_true", help="Generate the missing ones.")
    ap.add_argument("--ranks", action="store_true", help="Recompute ranks (~12 min of CPU).")
    args = ap.parse_args()
    if args.hypotheses:
        return generate()

    import witness_set as ws

    cases = witnesses_by_case()
    hyp = load_hypotheses()
    pools = ws._pool_ids(POOL)
    ranks = (
        compute_ranks(cases)
        if args.ranks or not RANKS.is_file()
        else json.loads(RANKS.read_text(encoding="utf-8"))
    )

    # A witness with no hypotheses for its case cannot be simulated. It is counted as
    # unsimulatable rather than assumed unreachable -- an absent measurement is not a
    # negative one, and imputing here would understate every cut equally.
    rows: list[tuple[str, list[str], bool, int | None]] = []
    for case, items in cases.items():
        if case not in pools:
            continue
        for pid, sources in items:
            pooled = pid in pools[case]
            r = ranks.get(case, {}).get(pid)
            rows.append((pid, sources, pooled, r))

    # A witness with no rank and no pool membership is not "unsimulatable" — it is
    # UNREACHABLE BY THIS CHANNEL, at any cut. 121 of 122 are non-arXiv ids: a dense index of
    # arXiv abstracts cannot return a Europe PMC or OpenAlex paper however wide the cut. That
    # is a ceiling on what widening can buy, and it belongs in the artifact next to the curve
    # rather than as a footnote — reading 0.67 at cut 10,000 without it invites "why not 1.0".
    reachable = [r for r in rows if r[3] is not None or r[2]]
    unreachable = [r for r in rows if r[3] is None and not r[2]]

    def reach_at(cut: int | None) -> dict[str, Any]:
        """cut=None is the status quo: pool membership as collected."""
        hit = [r for r in rows if r[2] or (cut is not None and r[3] is not None and r[3] < cut)]
        per_source: dict[str, dict[str, int]] = {}
        for _pid, sources, pooled, rank in rows:
            got = pooled or (cut is not None and rank is not None and rank < cut)
            for s in sources:
                d = per_source.setdefault(s, {"n": 0, "reached": 0})
                d["n"] += 1
                d["reached"] += got
        return {
            "n": len(rows),
            "reached": len(hit),
            "p": round(len(hit) / len(rows), 4),
            "by_source": {
                s: {"n": d["n"], "reached": d["reached"], "p": round(d["reached"] / d["n"], 4)}
                for s, d in sorted(per_source.items())
            },
        }

    actual = reach_at(None)  # the pool as collected, for context
    curve = {str(c): reach_at(c) for c in CUTS}
    # THE LIKE-FOR-LIKE BASELINE is the simulation at the SHIPPED cut, not the collected pool.
    # The two differ by 0.058, and that difference is not the cut. `rr_hyde_hypotheses` is a
    # POOL_FLAG because hypotheses are LLM output regenerated per collection, so the pool used
    # a different draw than this simulation does. The excess is uniform across cases whose
    # hypotheses were cached (6.0%) and freshly generated (5.4%) -- which is what a draw effect
    # looks like and is not what "old versus new hypotheses" would look like. Index drift is
    # excluded: the shards were last written 2026-08-06 and the pool collected 2026-08-20.
    # Measuring the cut against the collected pool would bill a hypothesis redraw as widening.
    base = curve[str(SHIPPED_CUT)]
    at_bar = curve[str(1000)]["p"]

    out: dict[str, Any] = {
        "_comment": (
            "NR-46 / item 10 stage 1: witness reach simulated at wider HyDE cuts, without "
            "re-collecting any pool. A witness counts as reached at cut K if it is already in "
            "the frozen pool or its HyDE rank is below K. $0 apart from the one-time "
            "hypothesis generation. Derived by evals/hyde_cut_reach.py; pinned by "
            "tests/test_hyde_cut_reach.py. This bounds reach from BELOW and decides only "
            "whether a paid arm is worth buying -- it cannot model the ranker renormalising "
            "or gate_depth still showing the gate 50 of a larger pool, which is where NR-11's "
            "damage lived."
        ),
        "pre_registered": {
            "bar_at_1000": BAR,
            "kill_at_1000": KILL,
            "baseline_reach": base["p"],
            "written_before_running": True,
        },
        "pool": POOL,
        "shipped_cut": SHIPPED_CUT,
        "cases_with_hypotheses": len(hyp),
        "cases_measured": len({c for c in cases if c in pools}),
        "witnesses": {
            "total_non_self": len(rows),
            "reachable_by_hyde": len(reachable),
            "unreachable_by_hyde": len(unreachable),
            "unreachable_non_arxiv": sum(1 for r in unreachable if not is_arxiv_id(r[0])),
            "unreachable_arxiv_not_indexed": sum(1 for r in unreachable if is_arxiv_id(r[0])),
            "hyde_ceiling": round(len(reachable) / len(rows), 4),
            "_comment": (
                "The ceiling is what widening cannot pass: a dense index of arXiv abstracts "
                "cannot return a non-arXiv paper at any cut, and 121 of the 122 unreachable "
                "witnesses are DOIs. Reach at cut 10,000 is 87.7% OF THIS CEILING, not of 1.0."
            ),
        },
        "pool_as_collected": actual,
        "baseline_simulated_at_shipped_cut": base,
        "hypothesis_draw_effect": {
            "_comment": (
                "Simulated reach at the shipped cut minus the pool's actual reach. Same cut, "
                "different hypothesis draw, so this is what redrawing hypotheses is worth -- "
                "measured by accident and worth its own line. C-7's rule applies: a single "
                "draw's level is not a property of the method."
            ),
            "actual": actual["p"],
            "simulated_same_cut": base["p"],
            "delta": round(base["p"] - actual["p"], 4),
        },
        "curve": curve,
    }
    out["verdict"] = {
        "reach_at_1000": at_bar,
        "passes_bar": bool(at_bar >= BAR),
        "killed": bool(at_bar < KILL),
        "marginal": bool(KILL <= at_bar < BAR),
        "baseline": base["p"],
        "baseline_is": "simulated at the shipped cut, same hypotheses — not the collected pool",
        "hyde_ceiling": round(len(reachable) / len(rows), 4),
        "share_of_ceiling_at_1000": round(at_bar / (len(reachable) / len(rows)), 4),
        "absolute_gain": round(at_bar - base["p"], 4),
        "relative_gain": round((at_bar - base["p"]) / base["p"], 3) if base["p"] else None,
    }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\npool as collected:            {actual['reached']}/{actual['n']} = {actual['p']:.4f}")
    print(
        f"BASELINE simulated at cut {SHIPPED_CUT}: "
        f"{base['reached']}/{base['n']} = {base['p']:.4f}   "
        f"(hypothesis-draw effect {base['p'] - actual['p']:+.4f})"
    )
    print(f"{'cut':>8}{'reached':>10}{'p':>9}{'gain':>9}")
    for c in CUTS:
        v = curve[str(c)]
        print(f"{c:>8}{v['reached']:>10}{v['p']:>9.4f}{v['p'] - base['p']:>+9.4f}")
    print(f"\nverdict: {json.dumps(out['verdict'])}")
    print(f"wrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
