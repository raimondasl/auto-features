"""Where do Opus 5's winning papers die in our pipeline, and how far below the cut? [NR-45]

P26 left the comparator question at +1.52 on the core 25 and -3.17 on materials science, with
the two systems level on the 32 of 37 repositories where Opus 5 does not over-answer. This asks
the operational follow-up: of the papers Opus 5 finds and the judge calls actionable, *which
stage of ours loses them*. Every input is already on disk.

**Stage 1, the funnel.** Each of Opus 5's 302 judged-actionable picks is walked through the
shipped pipeline: is it arXiv at all -> is it in our 3.1M dense index -> did it reach our
candidate pool -> did we show it. The stage where a paper dies names the subsystem that loses
it.

    non-arXiv                30.5%    structurally out of the shipped configuration's reach
    not in our index          0.3%    ONE paper of 302
    in index, never pooled   54.3%    <- the lever
    pooled, not shown        10.6%
    we showed it too          4.3%

**P12 holds against the comparator.** One paper in 302 is outside the index, so a wider corpus
buys essentially nothing -- the same verdict P12 reached against the gold set, now confirmed
against a frontier model's picks. And on materials science, the cohort we lose outright,
non-arXiv is **5.9%**: P24/P25's "no more sources" survives exactly where it would have been
most tempting to reopen. Bio is the mirror image at 68.1% non-arXiv, which is why Europe PMC's
+4.17 source term lives there and nowhere else.

**Stage 2, the rank probe.** `hyde.top_k = 100` candidates per hypothesis, four hypotheses,
the union feeds the ranker -- so the 164 stage-3 papers died at that cut. How far below it?

    median rank 1,087   p25 323   p75 3,562
    reachable at the shipped cut of 100:  11.5%
    at 1,000: 49.0%   at 2,000: 62.5%   at 5,000: 78.8%   at 10,000: 89.4%

**The union is too narrow; the hypotheses are not in the wrong register.** That was the fork
this probe existed to settle, and it settles it in the cheaper direction -- `top_k` is a config
integer, not a new mechanism.

**What this does NOT establish, and the reason the item it opens is not a patch.** Reach is not
net@2. NR-11 recorded a wider pool meeting a near-binary gate and making the headline *worse*,
and P4's pool expansion measured as a wash until the fine-scale rescore ranked what the gate
admitted (section 8.2's composition finding). Widening the cut is therefore a candidate to be
measured, not a fix to be applied, and its stage-1 evaluation is free: witness reach over the
520 non-self witnesses, where `cli-v2-opus5@30` currently sits at **0.142**.

    uv run python evals/opus5_funnel.py            # $0, funnel + cached ranks
    uv run python evals/opus5_funnel.py --ranks    # recompute ranks (~6 min of CPU)
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id, is_arxiv_id  # noqa: E402

RES = EVALS / "results"
INDEX = EVALS / ".work" / "hyde_index"
POOL = EVALS / ".work" / "pool-core25-arxiv"
HYPOTHESES = EVALS / ".work" / "hyde_hypotheses.json"
RANK_CACHE = EVALS / ".work" / "rank_probe_raw.json"
FROZEN = EVALS / "opus5_funnel.json"
CONTROL = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json"
SHIPPED_TOP_K = 100  # hyde.top_k — the cut these papers failed to clear
CUTS = (100, 200, 400, 1000, 2000, 5000, 10000, 50000)
STAGES = (
    "0_non_arxiv",
    "1_not_in_index",
    "2_index_but_not_pool",
    "3_pool_but_not_shown",
    "4_we_showed_it_too",
)


def cohort(case: str) -> str:
    return "bio" if case.startswith("bio-") else "mat" if case.startswith("mat-") else "core"


def load() -> tuple[dict, set, dict, dict]:
    rows = json.loads((EVALS / "gold_spread_v2_opus5.json").read_text(encoding="utf-8"))["results"]
    picks: dict[str, list[str]] = {}
    for key, r in rows.items():
        draw, case = key.split("/", 1)
        if draw == "1" and r.get("status") == "ok":
            picks[case] = [dedup_id(str(p)) for p, s in (r.get("scores") or {}).items() if s >= 2]
    index_ids = set()
    for f in sorted(INDEX.glob("*.ids")):
        index_ids.update(dedup_id(i) for i in f.read_text(encoding="utf-8").splitlines() if i)
    pool = {
        f.stem: {
            dedup_id(str(c["arxiv_id"]))
            for c in json.loads(f.read_text(encoding="utf-8"))["candidates"]
        }
        for f in sorted(POOL.glob("*.json"))
    }
    run = json.loads((RES / CONTROL).read_text(encoding="utf-8"))
    shown = {
        e["case"]: {dedup_id(str(p["arxiv_id"])) for p in e["returned"]["reporadar_toppicks"]}
        for e in run
    }
    return picks, index_ids, pool, shown


def stage_of(case, pid, index_ids, pool, shown) -> str:
    if not is_arxiv_id(pid):
        return "0_non_arxiv"
    if pid not in index_ids:
        return "1_not_in_index"
    if pid not in pool.get(case, set()):
        return "2_index_but_not_pool"
    if pid not in shown.get(case, set()):
        return "3_pool_but_not_shown"
    return "4_we_showed_it_too"


def compute_ranks(picks, index_ids, pool) -> dict[str, dict[str, int]]:
    """Rank of each stage-2 paper under the best of the case's four cached hypotheses.

    Uses the shipped distance function over the shipped index, and the hypotheses the
    replication froze, so the protocol cannot drift from the one the product runs.
    """
    import numpy as np

    from reporadar import hyde

    hyp = json.loads(HYPOTHESES.read_text(encoding="utf-8"))
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
    for case in sorted(picks):
        if case not in hyp:
            continue
        miss = [
            p for p in picks[case] if is_arxiv_id(p) and p in pos and p not in pool.get(case, set())
        ]
        if not miss:
            continue
        bits = hyde.encode_binary(model, list(hyp[case]))
        best = dict.fromkeys(miss, 10**9)
        for row in range(bits.shape[0]):
            d = np.concatenate(
                [hyde._hamming(np.load(s, mmap_mode="r"), bits[row]) for s in shards]
            )
            for p in miss:
                best[p] = min(best[p], int((d < d[pos[p]]).sum()))
        out[case] = best
        print(f"  {case:<12} {len(miss):>2} papers")
    RANK_CACHE.write_text(json.dumps(out, indent=0), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", action="store_true", help="Recompute ranks (~6 min of CPU).")
    args = ap.parse_args()

    picks, index_ids, pool, shown = load()
    per_stage: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for case, ps in picks.items():
        for pid in ps:
            st = stage_of(case, pid, index_ids, pool, shown)
            per_stage[case][st] += 1

    def funnel(cases) -> dict[str, Any]:
        agg: collections.Counter = collections.Counter()
        for c in cases:
            agg.update(per_stage[c])
        n = sum(agg.values())
        return {
            "n_cases": len(cases),
            "actionable_picks": n,
            **{k: agg[k] for k in STAGES},
            **{f"{k}_share": round(agg[k] / n, 3) for k in STAGES},
        }

    cases = sorted(picks)
    arm = json.loads((EVALS / "opus5_arm.json").read_text(encoding="utf-8"))["per_case"]
    losing = sorted(c for c, v in arm.items() if v["arxiv"] < v["opus5"])

    out: dict[str, Any] = {
        "_comment": (
            "NR-45: which stage of OUR pipeline loses the papers Opus 5 finds, and how far "
            "below the HyDE cut they sit. $0 -- stored picks, stored index, stored pools, "
            "stored runs. Derived by evals/opus5_funnel.py; pinned by "
            "tests/test_opus5_funnel.py. Reach is not net@2: NR-11 recorded a wider pool "
            "meeting a near-binary gate and making the headline worse, so the item this "
            "opens is a measurement, not a patch."
        ),
        "shipped_hyde_top_k": SHIPPED_TOP_K,
        "control_run": CONTROL,
        "funnel": {
            "all37": funnel(cases),
            "core": funnel([c for c in cases if cohort(c) == "core"]),
            "bio": funnel([c for c in cases if cohort(c) == "bio"]),
            "mat": funnel([c for c in cases if cohort(c) == "mat"]),
            "cases_we_lose": funnel([c for c in cases if c in losing]),
        },
    }

    ranks = (
        compute_ranks(picks, index_ids, pool)
        if args.ranks
        else (json.loads(RANK_CACHE.read_text(encoding="utf-8")) if RANK_CACHE.is_file() else {})
    )
    if ranks:
        flat = sorted(r for v in ranks.values() for r in v.values())
        n = len(flat)
        out["rank_probe"] = {
            "_comment": (
                "Rank of each stage-2 paper under the best of its case's four cached "
                "hypotheses, over the shipped index with the shipped distance function. "
                "Covers the cases whose hypotheses the replication froze; the rest (all six "
                "materials cases among them) would need fresh generation and are absent "
                "rather than assumed."
            ),
            "n_papers": n,
            "n_cases": len(ranks),
            "median": flat[n // 2],
            "p25": flat[n // 4],
            "p75": flat[3 * n // 4],
            "min": flat[0],
            "max": flat[-1],
            "recovered_at_cut": {str(c): sum(1 for r in flat if r < c) for c in CUTS},
            "share_at_cut": {str(c): round(sum(1 for r in flat if r < c) / n, 3) for c in CUTS},
            "verdict": {
                "union_too_narrow": True,
                "hypotheses_in_wrong_register": False,
                "reachable_at_shipped_cut": round(sum(1 for r in flat if r < SHIPPED_TOP_K) / n, 3),
            },
        }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out.get("funnel"), indent=1))
    if "rank_probe" in out:
        rp = out["rank_probe"]
        print(f"\nranks: median {rp['median']:,}  p75 {rp['p75']:,}  n={rp['n_papers']}")
        for c in CUTS:
            got = rp["recovered_at_cut"][str(c)]
            share = rp["share_at_cut"][str(c)]
            print(f"  cut {c:>6,}: {got:>3} ({share:.1%})")
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
