"""Does a citation hop from the repo's own bibliography reach the papers keyword search misses?

Free and keyless. **This is the only approach measured so far that reaches them at all**:
14/24, against 0/24 for keyword search and 2/24 for LLM-generated phrases.

    uv run python evals/diagnose_citation_hop.py

A dependency investigation reported 20/24 with *two* hops; one hop measures 14/24 here, so
two remains plausible and unverified. See RESULTS.md -> "Candidate-pool diagnosis".

Seeds: arXiv ids the repo itself cites (README, docs/, .bib, CITATION.cff) — the only seed
set a cold-start repo has, since the benchmark has no ratings or stars.

Two directions, measured separately:
  backward = references of the seeds   (what the repo's papers build on)
  forward  = papers citing the seeds   (later work that may supersede them)

Reports recall AND candidate-set size. Reaching a target inside a 30,000-paper set is not
reaching it — the shortlist still has to survive a Top-10 cut.
"""

import contextlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reporadar.citations import _s2_batch_post, _s2_id  # noqa: E402

TARGETS = {
    "rag": ["2409.14683", "2404.02805", "2501.17788", "2304.01982", "2505.11471"],
    "cv": ["1704.04503", "2012.07177", "2201.03545"],
    "rl": ["1509.06461", "1511.05952", "2110.02034"],
    "peft": ["2404.03592", "2405.12130"],
    "diffusion": ["2410.05317", "2508.16211"],
    "graph": ["2202.13013", "2303.06147", "2111.14522"],
    "speech": ["2303.00747", "2211.17192", "2311.00430"],
    "crypto": ["1812.04959", "2405.18993"],
    "systems": ["1512.00727"],
}

ID = re.compile(r"(?:arxiv\.org/abs/|arXiv[:/])(\d{4}\.\d{4,5})", re.I)


def seeds_for(case: str) -> list[str]:
    repo = Path(__file__).resolve().parent / ".work" / case
    found: set[str] = set()
    for pat in ("*.md", "*.rst", "*.cff", "*.bib"):
        for f in list(repo.rglob(pat))[:5000]:
            with contextlib.suppress(OSError, UnicodeDecodeError):
                found |= set(ID.findall(f.read_text(encoding="utf-8", errors="replace")))
    return sorted(found)


def hop(arxiv_ids: list[str], direction: str, cap: int = 60) -> set[str]:
    """One citation hop. `direction` is 'references' or 'citations'."""
    out: set[str] = set()
    ids = [_s2_id(a) for a in arxiv_ids[:cap]]
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        data = _s2_batch_post(chunk, f"{direction}.externalIds", None, 3, 3.0)
        if not data:
            continue
        for entry in data:
            for ref in (entry or {}).get(direction) or []:
                ax = ((ref or {}).get("externalIds") or {}).get("ArXiv")
                if ax:
                    out.add(ax.split("v")[0])
        time.sleep(3)
    return out


def main() -> int:
    rows = []
    for case, targets in TARGETS.items():
        seeds = seeds_for(case)
        if not seeds:
            print(f"[{case}] 0 seeds — A3 inapplicable (targets: {len(targets)})")
            rows.append(
                {"case": case, "seeds": 0, "targets": targets, "back": [], "fwd": [], "n": 0}
            )
            continue
        seed_set = set(seeds)
        print(f"[{case}] {len(seeds)} seeds", flush=True)

        back = hop(seeds, "references")
        fwd = hop(seeds, "citations")
        pool = (back | fwd) - seed_set
        hit_b = sorted(set(targets) & back)
        hit_f = sorted(set(targets) & fwd)
        hit = sorted(set(targets) & pool)
        print(
            f"        backward={len(back):6d}  forward={len(fwd):6d}  union={len(pool):6d}"
            f"   recovered={len(hit)}/{len(targets)}"
            f"{'  ' + ','.join(hit) if hit else ''}",
            flush=True,
        )
        rows.append(
            {
                "case": case,
                "seeds": len(seeds),
                "targets": targets,
                "back": hit_b,
                "fwd": hit_f,
                "n": len(pool),
            }
        )

    tot_t = sum(len(r["targets"]) for r in rows)
    tot_h = len({t for r in rows for t in set(r["back"]) | set(r["fwd"])})
    tot_n = sum(r["n"] for r in rows)
    print("\n=== A3 ONE-HOP VERIFICATION ===")
    print(f"targets: {tot_t}")
    print(f"recovered: {tot_h}  ({tot_h / tot_t:.0%})")
    print(f"total candidate papers produced: {tot_n}")
    if tot_h:
        print(f"needle-in-haystack ratio: 1 good paper per {tot_n // max(tot_h, 1):,} candidates")
    print("\ncompare: current TF-IDF queries 0/24 from 2030 fetched")
    out = Path(__file__).resolve().parent / ".work" / "diag_citation_hop.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
