"""P1 pass 1+2: persist the citation-hop pool with per-candidate coupling degrees.

The hop is the only discovery channel with measured recall — 18/24 against 0-3/24 for
everything else — and it is unusable as it stands: 92,014 candidates at a density of 1 good
paper per 5,111 (RESEARCH.md §3.5). The filter that would make it usable is unbuilt, and it
cannot be built because the pool has never been *persisted*: `diag_citation_hop.json` keeps
only summary counts, so every filter idea would need a fresh 20-minute network sweep to
evaluate. This writes the pool down once, with the free structural features attached, so the
sweep in `sweep_hop_filter.py` runs offline and instantly.

    uv run python evals/build_hop_pool.py              # all seeded cases
    uv run python evals/build_hop_pool.py --case peft  # one, for a smoke run

Free and keyless. Two passes, and the split is not incidental:

  pass 1  the hop itself, requesting only `{direction}.externalIds`, counting per-candidate
          forward/backward coupling degree. Nested payloads stay small, so the existing
          item-count truncation guard in `diagnose_citation_hop.hop` stays valid.
  pass 2  title/abstract/year/citationCount for the deduped ids, 500 per request. These are
          FLAT fields, so no nested cap applies.

Doing it in one pass — asking for `citations.title,citations.abstract` — would put ~11MB of
abstracts in a hub seed's response and hit S2's ~10MB body cap. That truncation is a *byte*
truncation, which lands BELOW the item-count guard and would therefore be silently accepted:
the exact shape of the bug that published a 14/24 (RESEARCH.md §6.5).

**Known bias, recorded rather than hidden.** 13 seeds across `rl` and `graph` saturate the
9,999 nested cap even when requested alone; the API's `offset + limit < 10000` wall means
nobody can enumerate them. Candidates reachable only through those seeds are undercounted,
so degrees are a *lower bound*. `truncated_seeds` per case is written to the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml  # noqa: E402
from diagnose_citation_hop import hop, seeds_for  # noqa: E402
from diagnose_pool import actionable_baseline_ids  # noqa: E402

from reporadar.citations import _s2_batch_post, _s2_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
OUT_DIR = EVALS / ".work" / "hop_pool"


def resolve_targets() -> dict[str, list[str]]:
    """{case: known-good arXiv ids} for EVERY benchmark case, derived not hardcoded.

    `diagnose_citation_hop.TARGETS` is a frozen literal covering the nine cohort-1 cases
    that had targets when the 18/24 result was measured. Reading it here meant every case
    added after that raised KeyError — which is how ten new cases silently failed to build
    until the traceback was noticed under a `grep` that was filtering it out.

    The canonical list is baseline picks intersected with judge score >= 2, exactly as
    `diagnose_pool` computes it. Same rule that made a stray `baseline_ids.json` dangerous:
    derive the target list, never read a copy of it.
    """
    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    return {c["name"]: actionable_baseline_ids(c["name"]) for c in bench["cases"]}


META_CACHE = OUT_DIR / "_meta_cache.json"
META_FIELDS = "externalIds,title,abstract,year,citationCount"

# Keyless S2 is a shared anonymous pool. A first attempt at 500 ids / 2s got HTTP 429 on
# most batches and filled 18% of one case. RESEARCH.md §3.4 records the other failure mode:
# sustained polling earned this machine a ~70-minute IP block. So: smaller chunks, a longer
# floor between requests, and far more patience per request than the interactive default.
# Request RATE is the lever, not page size.
META_CHUNK = 200
META_SLEEP = 5.0
META_RETRIES = 6
META_BACKOFF = 5.0


def load_meta_cache() -> dict[str, dict]:
    if META_CACHE.is_file():
        cached: dict[str, dict] = json.loads(META_CACHE.read_text(encoding="utf-8"))
        return cached
    return {}


def save_meta_cache(cache: dict[str, dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_CACHE.write_text(json.dumps(cache), encoding="utf-8")


def fetch_metadata(arxiv_ids: list[str], cache: dict[str, dict]) -> dict[str, dict]:
    """Fill *cache* with flat metadata for any of *arxiv_ids* it does not already hold.

    Resumable and shared across cases, which matters twice: the anonymous endpoint cannot
    deliver ~92k ids in one sitting, and candidate sets overlap between repos — an id
    fetched for `cv` is free for `rl`. Re-running the script tops up the gaps instead of
    starting over.

    A failed id is left ABSENT rather than stored as empty, so a later run retries it and
    no downstream filter mistakes "S2 did not answer" for "0 citations" (absent-is-not-zero,
    the rule the ranker already uses for upvotes).
    """
    todo = [a for a in arxiv_ids if a not in cache]
    if not todo:
        return cache
    print(f"    metadata: {len(todo)} missing of {len(arxiv_ids)}", flush=True)
    for start in range(0, len(todo), META_CHUNK):
        chunk = todo[start : start + META_CHUNK]
        data = _s2_batch_post(
            [_s2_id(a) for a in chunk], META_FIELDS, None, META_RETRIES, META_BACKOFF
        )
        time.sleep(META_SLEEP)
        if not data:
            print(f"    ! batch failed at {start}; {len(chunk)} left for a later run")
            continue
        for entry in data:
            if not entry:
                continue
            ax = ((entry.get("externalIds") or {}).get("ArXiv") or "").split("v")[0]
            if not ax:
                continue
            cache[ax] = {
                "title": entry.get("title") or "",
                "abstract": entry.get("abstract") or "",
                "year": entry.get("year"),
                "citation_count": entry.get("citationCount"),
            }
        done = min(start + META_CHUNK, len(todo))
        print(f"    metadata {done}/{len(todo)}  (cache {len(cache)})", flush=True)
        if done % (META_CHUNK * 5) == 0:
            save_meta_cache(cache)  # checkpoint, so a block mid-run loses at most 5 batches
    save_meta_cache(cache)
    return cache


def build_case(case: str, targets: list[str], *, with_metadata: bool = True) -> dict | None:
    seeds = seeds_for(case)
    if not seeds:
        print(f"[{case}] 0 seeds — no pool to build")
        return None
    print(f"[{case}] {len(seeds)} seeds; hopping...", flush=True)
    b_res = hop(seeds, "references")
    f_res = hop(seeds, "citations")
    back, fwd = b_res.reached, f_res.reached
    trunc_b, trunc_f = b_res.truncated_seeds, f_res.truncated_seeds
    failed = b_res.failed_chunks + f_res.failed_chunks
    if failed:
        # Refuse rather than persist a throttled undercount. The first build of this
        # pool silently recovered 11% of its known size and looked like a finding.
        print(f"    !! {failed} chunk(s) failed — REFUSING to write {case}; re-run later")
        return None

    seed_set = set(seeds)
    ids = sorted((set(back) | set(fwd)) - seed_set)
    print(f"    pool={len(ids)}  (backward {len(back)}, forward {len(fwd)})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta_cache()
    if with_metadata:
        meta = fetch_metadata(ids + seeds, meta)
    seed_years = [m["year"] for s in seeds if (m := meta.get(s)) and isinstance(m.get("year"), int)]
    with (OUT_DIR / f"{case}.jsonl").open("w", encoding="utf-8") as fh:
        for ax in ids:
            m = meta.get(ax, {})
            fh.write(
                json.dumps(
                    {
                        "id": ax,
                        "fwd_degree": fwd.get(ax, 0),
                        "back_degree": back.get(ax, 0),
                        "year": m.get("year"),
                        "citation_count": m.get("citation_count"),
                        "title": m.get("title", ""),
                        "abstract": m.get("abstract", ""),
                        "is_target": ax in targets,
                    }
                )
                + "\n"
            )
    row = {
        "case": case,
        "seeds": len(seeds),
        "pool": len(ids),
        "targets": targets,
        "targets_in_pool": sorted(set(targets) & set(ids)),
        "truncated_seeds": trunc_b + trunc_f,
        "failed_chunks": failed,
        # Coverage over THIS case's candidates. `len(meta)` would be the global shared
        # cache and could exceed the case pool, reporting >100%.
        "metadata_coverage": round(sum(1 for a in ids if a in meta) / max(len(ids), 1), 4),
        "seed_median_year": sorted(seed_years)[len(seed_years) // 2] if seed_years else None,
    }
    print(
        f"    wrote {len(ids)} rows; targets in pool {len(row['targets_in_pool'])}"
        f"/{len(targets)}; metadata coverage {row['metadata_coverage']:.0%}",
        flush=True,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="build one case only")
    ap.add_argument(
        "--skip-metadata",
        action="store_true",
        help="pass 1 only — hop and record degrees, leaving title/abstract/year to a "
        "later resumable run. Degrees are the signal P1 filters on and are complete "
        "after pass 1; metadata only adds the secondary year/citation dimensions.",
    )
    args = ap.parse_args()

    targets = resolve_targets()
    if args.case and args.case not in targets:
        print(f"unknown case {args.case!r}; benchmark has: {', '.join(sorted(targets))}")
        return 1
    cases = {args.case: targets[args.case]} if args.case else targets
    rows = [
        r for c, t in cases.items() if (r := build_case(c, t, with_metadata=not args.skip_metadata))
    ]
    if not rows:
        print("no pools built")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "manifest.json"
    existing = (
        {r["case"]: r for r in json.loads(manifest.read_text(encoding="utf-8"))}
        if manifest.is_file()
        else {}
    )
    existing.update({r["case"]: r for r in rows})
    merged = [existing[k] for k in sorted(existing)]
    manifest.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    tot_pool = sum(r["pool"] for r in merged)
    tot_hit = sum(len(r["targets_in_pool"]) for r in merged)
    print(f"\n=== {len(merged)} case(s) persisted ===")
    print(f"pool total {tot_pool:,}   targets in pool {tot_hit}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
