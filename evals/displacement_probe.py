"""How much of §21's displacement is the ranker's absent-category rule? ($0, offline, judge-free.)

    uv run python evals/displacement_probe.py

§21.4 measured that adding Europe PMC displaced **19 of the arXiv-only run's 43 Top Picks**, and
§21.7 filed that as competition for a fixed 15-slot window. Two facts found afterwards say the
framing was incomplete.

**The displaced papers were good.** 18 of 19 were judge-actionable (95%) — a *higher* rate than
the 24 that survived (88%). So the swap did not drop weak papers for strong ones; it dropped good
papers for other good papers, which is what a capacity limit looks like rather than a quality
improvement.

**And one side of that contest was carrying a handicap the codebase already documents.** Both
arms ran `absent_category='omit'`, the shipped default, and `ranker.score_paper`'s own comment
says what that does: an arXiv paper is averaged over keyword AND category while a paper with no
comparable category is averaged over keyword alone, so *at equal keyword relevance the
uncategorised paper scores higher* — 0.600 against 0.567, or 0.600 against 0.400 when the arXiv
paper matches no target category. Every Europe PMC paper is uncategorised by that test (§12.3
moved them onto exactly this path).

So §21's substitution may be partly a scoring rule rather than relevance, and this measures how
much. Ranking is deterministic given a pool, both pools are on disk, and `rank_candidates` is
separately callable — so the counterfactual costs nothing.

**Corrected 2026-08-21, at the unit.** The first version compared heuristic top-15s. That is not
a stage the product ships: `rank_candidates` produces the pre-gate ordering, and the shipped
window is `rerank_by_actionability(gated)[:15]` over the `--rr-pool` candidates. So the
absent-category rule governs **which papers reach the gate at all**, and the cut to compare is
the gate-entry depth. §23's kill-clause check caught it — the reconstruction failed to reproduce
the shipped ranks 1–15 on all six cases, which is precisely what that check exists for. The
finding is unchanged; the figures are not: **40%/40% at the gate-entry cut**, against the 51%/50%
first reported at the wrong one.

**What it cannot answer.** Whether the papers each mode prefers are *better*: that needs judging
the ones `omit` never showed, which is a paid arm and is not this. This reports composition and
rank movement only, and every number in it is judge-free.
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
POOL = WORK_DIR / "pool-epmc-treat"
MODES = ("omit", "impute", "zero")
# The GATE-ENTRY cut (`--rr-pool`), not `output.top_n`. Corrected 2026-08-21 after §23's
# kill-clause check: the shipped window is `rerank_by_actionability(...)[:15]` over the papers
# the heuristic ranking hands to the gate, so the absent-category rule governs *which papers are
# gated at all*, and comparing heuristic top-15s measured a stage the product does not ship.
DEFAULT_CUT = 50


def rank_under(
    case: str, papers: list[dict[str, Any]], categories: list[str], mode: str, cut: int
) -> list[str]:
    """The ids of the top-*cut* papers under one absent-category mode, best first."""
    ranked = rank_candidates(
        WORK_DIR / case,
        papers,
        categories,
        top_n=cut,
        all_time=True,
        hybrid=True,
        absent_category=mode,
    )
    return [p["arxiv_id"] for p, _score in ranked]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=",".join(BIO6))
    ap.add_argument("--cut", type=int, default=DEFAULT_CUT, help="gate-entry depth (--rr-pool)")
    ap.add_argument("--out", default=str(WORK_DIR / "displacement_probe.json"))
    args = ap.parse_args()

    bench = {c["name"]: c for c in load_benchmark()["cases"]}
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    out: dict[str, Any] = {"cut": args.cut, "cases": {}}

    print("=" * 78)
    print(f"EUROPE PMC SHARE OF THE TOP-{args.cut} GATE-ENTRY CUT, BY MODE (judge-free)")
    print("=" * 78)
    print(f"  {'case':16} " + "  ".join(f"{m:>16}" for m in MODES))
    totals = dict.fromkeys(MODES, 0)
    n_ranked = 0
    for case in cases:
        pool = json.loads((POOL / f"{case}.json").read_text(encoding="utf-8"))["candidates"]
        cats = bench[case].get("expected_categories") or []
        row: dict[str, Any] = {}
        cells = []
        for mode in MODES:
            ids = rank_under(case, pool, cats, mode, args.cut)
            epmc = [i for i in ids if not is_arxiv_id(i)]
            row[mode] = {"ids": ids, "n_epmc": len(epmc)}
            totals[mode] += len(epmc)
            cells.append(f"{len(epmc):2d}/{len(ids):2d} ({len(epmc) / len(ids):4.0%})")
        n_ranked += len(row["omit"]["ids"])
        out["cases"][case] = row
        print(f"  {case:16} " + "  ".join(f"{c:>16}" for c in cells))
    print(
        f"  {'TOTAL':16} "
        + "  ".join(
            f"{f'{totals[m]:2d}/{n_ranked:2d} ({totals[m] / n_ranked:4.0%})':>16}" for m in MODES
        )
    )

    print("\n" + "=" * 78)
    print("WHAT CHANGES WHEN THE RULE CHANGES — against the shipped 'omit' ranking")
    print("=" * 78)
    for mode in MODES[1:]:
        held = swapped = 0
        for case in cases:
            a = set(out["cases"][case]["omit"]["ids"])
            b = set(out["cases"][case][mode]["ids"])
            held += len(a & b)
            swapped += len(a - b)
        print(
            f"  {mode:8} keeps {held:3d} of the {n_ranked} shipped gate-entry slots, "
            f"replaces {swapped:3d}"
        )

    share_omit = totals["omit"] / n_ranked
    share_impute = totals["impute"] / n_ranked
    print(
        f"\n  'omit' is shipped and gives Europe PMC {share_omit:.0%} of the GATE-ENTRY cut —\n"
        "  the stage this rule actually governs. 'impute', which scores a missing category at the\n"
        f"  pool's own mean instead of dropping the term, gives it {share_impute:.0%}. The\n"
        "  hypothesis this probe was built to test was that the shipped rule inflates that\n"
        "  share. It does not: the two\n"
        "  agree to within a slot, so §21.4's displacement is NOT an artefact of the\n"
        "  absent-category rule and §21.7's capacity framing stands.\n\n"
        "  'zero' is the outlier, and it is not the principled option — it asserts that a bioRxiv\n"
        "  paper has zero topical match, when what it actually has is a different taxonomy.\n\n"
        "  This probe does NOT say which ranking is better: that needs judging papers the shipped\n"
        "  ranking never showed, which is a paid arm and needs its own pre-registration."
    )
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
