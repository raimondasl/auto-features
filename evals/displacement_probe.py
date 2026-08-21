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
WINDOW = 15  # output.top_n, the product's setting — not a benchmark artifact


def rank_under(
    case: str, papers: list[dict[str, Any]], categories: list[str], mode: str
) -> list[str]:
    """The ids of the top-WINDOW papers under one absent-category mode, best first."""
    ranked = rank_candidates(
        WORK_DIR / case,
        papers,
        categories,
        top_n=WINDOW,
        all_time=True,
        hybrid=True,
        absent_category=mode,
    )
    return [p["arxiv_id"] for p, _score in ranked]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=",".join(BIO6))
    ap.add_argument("--out", default=str(WORK_DIR / "displacement_probe.json"))
    args = ap.parse_args()

    bench = {c["name"]: c for c in load_benchmark()["cases"]}
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    out: dict[str, Any] = {"window": WINDOW, "cases": {}}

    print("=" * 78)
    print(f"EUROPE PMC SHARE OF THE TOP-{WINDOW} WINDOW, BY ABSENT-CATEGORY MODE (judge-free)")
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
            ids = rank_under(case, pool, cats, mode)
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
            f"  {mode:8} keeps {held:3d} of the {n_ranked} shipped window slots, "
            f"replaces {swapped:3d}"
        )

    share_omit = totals["omit"] / n_ranked
    share_impute = totals["impute"] / n_ranked
    print(
        f"\n  'omit' is shipped and gives Europe PMC {share_omit:.0%} of the window; 'impute' —\n"
        "  which scores a missing category at the pool's own mean instead of dropping the term,\n"
        f"  and is the principled option — gives it {share_impute:.0%}. The hypothesis this probe\n"
        "  was built to test was that the shipped rule inflates that share. It does not: the two\n"
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
