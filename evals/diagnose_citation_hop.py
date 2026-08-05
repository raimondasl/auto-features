"""Does a citation hop from the repo's own bibliography reach the papers keyword search misses?

Free and keyless. **This is the only approach measured so far that reaches them at all**:
18/24, against 0/24 for keyword search and 2/24 for LLM-generated phrases.

    uv run python evals/diagnose_citation_hop.py

An earlier version of this script recorded 14/24. That was a transport artifact, not a
result: it sent 100 seeds per request, and the batch endpoint truncates nested items at
9,999 across a request, filled greedily in id order — so one hub seed consumed the whole
budget and every later seed came back empty, HTTP 200, no error. `graph` scored 0/3 purely
because it has the most seeds. See RESULTS.md -> "Candidate-pool diagnosis".

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
from collections import Counter
from pathlib import Path
from typing import NamedTuple

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


_NESTED_CAP = 9999
_CHUNK = 4
_MAX_DEPTH = 3
# Keyless S2 is a shared anonymous pool and throttles hard. These are deliberately more
# patient than an interactive default: a chunk that exhausts its retries is DATA LOSS,
# and it used to be invisible (see HopResult.failed_chunks).
_RETRIES = 6
_BACKOFF = 5.0
_SLEEP = 3.0


class HopResult(NamedTuple):
    """What one hop produced, and everything that could make it an undercount.

    ``failed_chunks`` exists because it was missing and that cost a whole run. Chunks
    whose requests exhausted their retries used to be dropped with a bare ``return``, so
    throttling produced a smaller pool and no error: a rebuild recovered 10,374 of the
    known 92,014 candidates — 11% — with `diffusion` and `speech` at exactly zero, and
    reported success. A caller that ignores this field is reading fiction.
    """

    reached: Counter[str]
    truncated_seeds: int
    failed_chunks: int


def hop(arxiv_ids: list[str], direction: str, cap: int = 60) -> HopResult:
    """One citation hop. Returns a HopResult; check `failed_chunks` before using it.

    The count is the **coupling degree**: how many of the repo's own seeds co-cite (forward)
    or are cited by (backward) this candidate. It is the free structural signal P1 filters
    on, so it is collected here rather than in a second implementation — this function owns
    the truncation guard below, and a copy of that guard is exactly how a 14/24 got
    published (RESEARCH.md §6.5).

    A Counter is a drop-in for the set this used to return: `len()` and `in` and iteration
    all behave the same over its keys.

    **Chunking is the whole correctness story here.** The batch endpoint truncates nested
    items at 9,999 across a request, filled greedily in id order. The first version of this
    script sent 100 seeds per request and got `[9999, 0, 0, ...]` back — one seed's
    citations and every later seed blank, HTTP 200, no error. That understated recall and,
    worse, made it *decrease* with seed count, since a hub seed eats the whole budget.

    So: small chunks, and split on a capped response. A seed that still saturates alone is
    genuinely un-enumerable (a paper with ~40k citers cannot be paged past the API's global
    `offset + limit < 10000` wall) and is counted as truncated rather than silently
    accepted.
    """
    out: Counter[str] = Counter()
    truncated = 0
    failed = 0
    ids = [_s2_id(a) for a in arxiv_ids[:cap]]

    def collect(chunk: list[str], depth: int = 0) -> None:
        nonlocal truncated, failed
        data = _s2_batch_post(chunk, f"{direction}.externalIds", None, _RETRIES, _BACKOFF)
        time.sleep(_SLEEP)
        if not data:
            failed += 1  # never silent: this chunk's seeds contributed nothing
            return
        nested = sum(len((e or {}).get(direction) or []) for e in data)
        if nested >= _NESTED_CAP and len(chunk) > 1 and depth < _MAX_DEPTH:
            mid = len(chunk) // 2
            collect(chunk[:mid], depth + 1)
            collect(chunk[mid:], depth + 1)
            return
        if nested >= _NESTED_CAP:
            truncated += len(chunk)
        # S2 returns results positionally aligned with the posted ids, so each entry's
        # nested list belongs to exactly one seed. Dedupe within a seed before counting,
        # so a candidate cited twice by one seed still scores degree 1 for it.
        for entry in data:
            per_seed = {
                ax.split("v")[0]
                for ref in (entry or {}).get(direction) or []
                if (ax := ((ref or {}).get("externalIds") or {}).get("ArXiv"))
            }
            out.update(per_seed)

    for i in range(0, len(ids), _CHUNK):
        collect(ids[i : i + _CHUNK])
    return HopResult(out, truncated, failed)


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

        b_res = hop(seeds, "references")
        f_res = hop(seeds, "citations")
        back, fwd = b_res.reached, f_res.reached
        trunc_b, trunc_f = b_res.truncated_seeds, f_res.truncated_seeds
        failed = b_res.failed_chunks + f_res.failed_chunks
        if failed:
            print(
                f"        !! {failed} request chunk(s) FAILED — undercount, not a result",
                flush=True,
            )
        pool = (set(back) | set(fwd)) - seed_set
        hit_b = sorted(set(targets) & set(back))
        hit_f = sorted(set(targets) & set(fwd))
        hit = sorted(set(targets) & pool)
        note = f"  [{trunc_b + trunc_f} seed(s) un-enumerable]" if trunc_b + trunc_f else ""
        print(
            f"        backward={len(back):6d}  forward={len(fwd):6d}  union={len(pool):6d}"
            f"   recovered={len(hit)}/{len(targets)}"
            f"{'  ' + ','.join(hit) if hit else ''}{note}",
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
                "truncated_seeds": trunc_b + trunc_f,
                "failed_chunks": failed,
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
