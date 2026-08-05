"""How far does the citation hop reach across the WHOLE benchmark, not just where it works?

The published figure is 18/24 — the hop's recall over the nine cohort-1 cases that had
targets. That denominator quietly excluded every repo the hop cannot serve at all. With the
benchmark at 22 cases and 48 targets, the honest question is what fraction of *all* known-good
papers a bibliography-seeded hop can reach, and what predicts the difference.

    uv run python evals/hop_reach.py     # instant, offline, free

Seeds come from the repo's own docs (README, docs/, .bib, CITATION.cff), which is the only
seed set a cold-start repo has. A repo that cites no arXiv paper has no seeds and the hop is
structurally zero there — not merely bad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from build_hop_pool import OUT_DIR, resolve_targets  # noqa: E402
from diagnose_citation_hop import seeds_for  # noqa: E402

EVALS = Path(__file__).resolve().parent


def main() -> int:
    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    targets = resolve_targets()
    manifest = {}
    mf = OUT_DIR / "manifest.json"
    if mf.is_file():
        manifest = {r["case"]: r for r in json.loads(mf.read_text(encoding="utf-8"))}

    rows = []
    for c in bench["cases"]:
        name = c["name"]
        tg = targets.get(name, [])
        if not tg:
            continue  # no known-good papers -> nothing for the hop to reach
        seeds = len(seeds_for(name))
        m = manifest.get(name)
        rows.append(
            {
                "case": name,
                "seeds": seeds,
                "pool": m["pool"] if m else 0,
                "found": len(m["targets_in_pool"]) if m else 0,
                "targets": len(tg),
            }
        )

    print(f"{'case':11} {'seeds':>6} {'pool':>9} {'reached':>9}")
    for r in sorted(rows, key=lambda r: -r["seeds"]):
        note = "" if r["seeds"] else "   <- no bibliography: structurally 0"
        print(f"{r['case']:11} {r['seeds']:6} {r['pool']:9,} {r['found']}/{r['targets']:<7}{note}")

    tot_t = sum(r["targets"] for r in rows)
    tot_f = sum(r["found"] for r in rows)
    seeded = [r for r in rows if r["seeds"]]
    st, sf = sum(r["targets"] for r in seeded), sum(r["found"] for r in seeded)
    print(f"\nALL cases with targets   : {tot_f}/{tot_t} = {tot_f / tot_t:.0%}")
    print(
        f"only bibliography-seeded : {sf}/{st} = {sf / st:.0%} "
        f" ({len(seeded)} of {len(rows)} cases)"
    )
    print(f"unreachable by construction: {tot_t - st} targets in {len(rows) - len(seeded)} cases")

    # Does a bigger bibliography buy reach? Split at the median seed count.
    if len(seeded) >= 4:
        med = sorted(r["seeds"] for r in seeded)[len(seeded) // 2]
        hi = [r for r in seeded if r["seeds"] >= med]
        lo = [r for r in seeded if r["seeds"] < med]
        for label, grp in (("seeds >= median", hi), ("seeds <  median", lo)):
            t = sum(r["targets"] for r in grp)
            f = sum(r["found"] for r in grp)
            if t:
                print(f"  {label} ({med}): {f}/{t} = {f / t:.0%} over {len(grp)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
