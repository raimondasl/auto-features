"""§24.5: does adding a source reorder the papers that were already there? ($0, judge-free.)

    uv run python evals/rank_stability.py

§21.4 measured that adding Europe PMC displaced 19 of the arXiv-only run's 43 Top Picks, and
§24.5 found that 13 of those 19 fell **below rank 30** in a 30-deep re-run. Extending that here:
of the control arm's 90 top-15 papers, **40 are outside the treatment arm's top 30**.

None of that is a defect on its own. Europe PMC wins about 40% of the gate-entry cut (§22.2), so
arXiv papers must move down; a fixed window with more competitors shows fewer of them. That is
arithmetic, not a bug.

**The question that can distinguish a bug from arithmetic is different: does adding a source
change the order of the papers that were already there?** Whether one arXiv paper outranks
another is a statement about those two papers and the repository. A bioRxiv preprint arriving in
the pool is irrelevant to it, and a ranker that lets the newcomer change that verdict is
violating independence of irrelevant alternatives — which is a real defect, not a consequence of
a full window.

There are concrete mechanisms by which it could happen here, which is why it is worth measuring
rather than assuming: BM25 is corpus-relative so new documents move every IDF; RRF fuses *ranks*,
which all shift when documents are inserted; and `absent_category: impute` scores a missing
category at the **pool mean**, which the new papers change by definition.

**This measures the ranking stage, not the shipped window.** §22.2 was published at the wrong
unit for exactly this reason: `rank_candidates` returns the pre-gate ordering, and the shipped
window is `rerank_by_actionability(gated)[:15]`. Here the pre-gate ordering is the right object,
because the question is about the ranker.
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

from harness import WORK_DIR, load_benchmark  # noqa: E402
from run_judge_eval import rank_candidates  # noqa: E402

from reporadar.paper_id import is_arxiv_id  # noqa: E402

BIO6 = ("bio-align", "bio-singlecell", "bio-scvi", "bio-mdsim", "bio-mdtraj", "bio-kmer")
CONTROL_POOL = WORK_DIR / "pool-epmc-control"
TREAT_POOL = WORK_DIR / "pool-epmc-treat"
DEEP = 10_000  # rank the whole pool; the question is about order, not about a cut


def ranked_ids(case: str, pool_dir: Path, categories: list[str], hybrid: bool = True) -> list[str]:
    pool = json.loads((pool_dir / f"{case}.json").read_text(encoding="utf-8"))["candidates"]
    ranked = rank_candidates(
        WORK_DIR / case, pool, categories, top_n=DEEP, all_time=True, hybrid=hybrid
    )
    return [p["arxiv_id"] for p, _s in ranked]


def kendall_tau(a: list[str], b: list[str]) -> tuple[float, int, int]:
    """Kendall's tau-a over the shared items, plus the discordant/total pair counts.

    Written out rather than pulled from scipy because the eval extras do not carry scipy and
    a dependency for one statistic is not worth a lockfile change.
    """
    pos_b = {x: i for i, x in enumerate(b)}
    common = [x for x in a if x in pos_b]
    ranks = [pos_b[x] for x in common]
    n = len(ranks)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ranks[i] < ranks[j]:
                conc += 1
            elif ranks[i] > ranks[j]:
                disc += 1
    total = conc + disc
    return ((conc - disc) / total if total else 1.0), disc, total


def isolate(cases: list[str], bench: dict[str, Any]) -> None:
    """The same comparison with hybrid RRF off, which localises the cause.

    With `hybrid=False` and the shipped `absent_category: omit`, a paper's score depends only on
    that paper and the profile — nothing about the pool enters it. A well-behaved ranker must
    therefore return tau exactly 1.000 here, and any departure would mean the instability lives
    somewhere other than the fusion.
    """
    print("\n" + "=" * 92)
    print("ISOLATING THE CAUSE — hybrid RRF OFF, where the score is per-paper by construction")
    print("=" * 92)
    taus = []
    for case in cases:
        cats = bench[case].get("expected_categories") or []
        ctrl = ranked_ids(case, CONTROL_POOL, cats, hybrid=False)
        treat = [i for i in ranked_ids(case, TREAT_POOL, cats, hybrid=False) if is_arxiv_id(i)]
        tau, disc, total = kendall_tau(ctrl, treat)
        taus.append(tau)
        print(f"  {case:16} tau {tau:.4f}   discordant {disc}/{total}")
    print(f"  mean {sum(taus) / len(taus):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=",".join(BIO6))
    ap.add_argument("--isolate", default="bio-align,bio-scvi,bio-mdtraj")
    args = ap.parse_args()
    bench = {c["name"]: c for c in load_benchmark()["cases"]}
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    print("=" * 92)
    print("DO THE arXiv PAPERS KEEP THEIR ORDER WHEN A SECOND SOURCE IS ADDED? (judge-free)")
    print("=" * 92)
    print(f"  {'case':16} {'shared':>7} {'tau':>7} {'discordant pairs':>18}  {'top-15 kept':>12}")
    out: dict[str, Any] = {}
    taus, kept_tot, n_tot = [], 0, 0
    for case in cases:
        cats = bench[case].get("expected_categories") or []
        ctrl = ranked_ids(case, CONTROL_POOL, cats)
        treat = [i for i in ranked_ids(case, TREAT_POOL, cats) if is_arxiv_id(i)]
        tau, disc, total = kendall_tau(ctrl, treat)
        # Of the control's own top 15, how many are still in the treatment arm's arXiv top 15?
        top_t = set(treat[:15])
        kept = sum(1 for i in ctrl[:15] if i in top_t)
        taus.append(tau)
        kept_tot += kept
        n_tot += 15
        out[case] = {"tau": tau, "discordant": disc, "pairs": total, "top15_kept": kept}
        print(
            f"  {case:16} {len(set(ctrl) & set(treat)):7d} {tau:7.3f} "
            f"{disc:9d}/{total:<8d} {kept:9d}/15"
        )
    mean_tau = sum(taus) / len(taus)
    print(
        f"\n  mean Kendall tau {mean_tau:.3f}   control top-15 still in the arXiv top-15: "
        f"{kept_tot}/{n_tot} ({kept_tot / n_tot:.0%})"
    )
    print(
        "\n  tau = 1.000 would mean the newcomer changed nothing about how the incumbents\n"
        "  compare to each other, which is what a well-behaved ranker should do. Anything\n"
        "  materially below that is the ranker letting an irrelevant alternative decide\n"
        "  between two papers it has no bearing on.\n\n"
        "  This says nothing about which order is BETTER — that needs labels the shipped\n"
        "  ranking never produced, and it is not this."
    )
    isolate([c.strip() for c in args.isolate.split(",") if c.strip()], bench)

    dest = WORK_DIR / "rank_stability.json"
    dest.write_text(json.dumps({"mean_tau": mean_tau, "cases": out}, indent=1), encoding="utf-8")
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
