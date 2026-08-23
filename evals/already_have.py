"""§28 secondary + tertiary: does the repo already citing a paper predict that it is a dud? ($0.)

    uv run python evals/already_have.py

**The tertiary first, because it can make the primary moot.** §28.3 found that the eval harness
never applies `profiler.cited_arxiv_ids_of` while the product does (`cli.py:741`,
`profiler.py:939`). Every benchmark number in this project therefore describes a pipeline that
shows papers a real user would never be offered. So before asking *why* the gate over-admits at
score 3, ask how much of that over-admission a user ever sees.

**H-A (§28.1)**: the misfires are papers describing capability the repository already has — its
own paper, its predecessor, the method it implements. `cited_arxiv_ids_of` is the shipped, narrow
detector for that: a paper the repo cites in its README, `CITATION*` or `docs/` is one it
demonstrably knows about. It is run **as-is, with no tuning**; §9.0 measured that it misses
`scvi-tools` and `mace` because neither cites itself, and that incompleteness is a property under
test rather than a defect to patch first.

The membership rule is the product's own — `dedup_id(paper_id) in cited_ids`, `digest.py:244` —
rather than a second implementation of version-matching, which is the defect class this project
has paid for three times.

Both endpoints here are **$0**: every label is bought and every checkout is on disk.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from harness import WORK_DIR  # noqa: E402
from label_pool import fisher_exact  # noqa: E402
from score3_mechanism import collect  # noqa: E402
from second_judge import ACTIONABLE, DEFAULT_MODEL, second_cache_path  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.profiler import cited_arxiv_ids_of  # noqa: E402


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def compare(label: str, a: list[dict[str, Any]], b: list[dict[str, Any]], field: str) -> None:
    ka, na = sum(1 for r in a if r.get(field)), len(a)
    kb, nb = sum(1 for r in b if r.get(field)), len(b)
    if not na or not nb:
        print(f"  {label:30} one arm empty ({na} vs {nb}) — not reported")
        return
    la, ha = wilson(ka, na)
    lb, hb = wilson(kb, nb)
    print(
        f"  {label:30} {ka:2d}/{na:3d} = {ka / na:.3f} [{la:.3f},{ha:.3f}]   vs   "
        f"{kb:2d}/{nb:3d} = {kb / nb:.3f} [{lb:.3f},{hb:.3f}]   gap {ka / na - kb / nb:+.3f}  "
        f"p={fisher_exact(ka, na - ka, kb, nb - kb):.4f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    rows, _stats = collect()
    cited_by_case: dict[str, frozenset[str]] = {}
    for r in rows:
        case = r["case"]
        if case not in cited_by_case:
            cited_by_case[case] = cited_arxiv_ids_of(WORK_DIR / case)
        # The PRODUCT's rule, not a second copy of it (digest.py:244).
        r["cited"] = dedup_id(str(r["arxiv_id"])) in cited_by_case[case]
        p = second_cache_path(args.model, case, r["arxiv_id"])
        if p.is_file():
            r["sonnet_non_actionable"] = (
                int(json.loads(p.read_text(encoding="utf-8"))["score"]) < ACTIONABLE
            )

    bad = [r for r in rows if r["non_actionable"]]
    cited = [r for r in rows if r["cited"]]
    print(f"population: {len(rows)} score-3 papers over {len(cited_by_case)} repositories")
    print(
        f"  repositories citing any arXiv id: "
        f"{sum(1 for v in cited_by_case.values() if v)}/{len(cited_by_case)}"
    )
    print(f"  score-3 papers the repo already cites: {len(cited)}")

    print("\n" + "=" * 96)
    print("TERTIARY — how much of the measured score-3 problem does the PRODUCT already remove?")
    print("=" * 96)
    removed = [r for r in bad if r["cited"]]
    print(f"  non-actionable score-3 papers: {len(bad)}")
    print(
        f"  of those, already cited by the repo (so suppressed in the product): "
        f"{len(removed)}  ({len(removed) / len(bad):.0%})"
    )
    for r in removed:
        print(f"    - {r['case']:16} {r['title'][:62]}")
    kept = [r for r in bad if not r["cited"]]
    print(f"\n  reaching a real user anyway: {len(kept)}")
    for r in kept[:8]:
        print(f"    - {r['case']:16} {r['title'][:62]}")
    print(
        "\n  This is a benchmark-versus-product number, not a repair. The eval harness does not\n"
        "  apply the rule (§28.3), so the benchmark's score-3 problem is larger than the one a\n"
        "  user has by exactly this much."
    )

    print("\n" + "=" * 96)
    print("SECONDARY (H-A) — does the repo already citing a paper predict it is non-actionable?")
    print("=" * 96)
    uncited = [r for r in rows if not r["cited"]]
    compare("GPT-5.5: cited vs not", cited, uncited, "non_actionable")
    have = [r for r in rows if "sonnet_non_actionable" in r]
    compare(
        "Sonnet: cited vs not",
        [r for r in have if r["cited"]],
        [r for r in have if not r["cited"]],
        "sonnet_non_actionable",
    )
    print(
        "\n  §28.6's bar is a >= 20-point gap holding under both judges, and a WIN would license\n"
        "  a HELD-OUT confirmation only — never a repair. §28.4: these data generated the\n"
        "  hypothesis."
    )

    dest = WORK_DIR / "already_have.json"
    dest.write_text(
        json.dumps({"n": len(rows), "n_bad": len(bad), "n_removed": len(removed)}, indent=1),
        encoding="utf-8",
    )
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
