"""Fill title/abstract/year/citations for the hop pool — P2's prerequisite, resumably.

P2 matches gap-phrases against candidate *text*, so it needs abstracts that pass 1 does not
fetch. Keyless Semantic Scholar cannot deliver ~105k of them in one sitting (RESEARCH.md
§3.4: sustained polling earned this machine a ~70-minute IP block), so this is a separate,
interruptible, resumable job rather than part of the build.

    uv run python evals/fill_pool_metadata.py            # P1-filtered survivors + all targets
    uv run python evals/fill_pool_metadata.py --all      # the whole 105k pool

**Default is the filtered set, and that is a measurement decision, not just thrift.** P1's
cut (`fwd>=2 OR back>=3`, cross-repo df<=2) keeps 16/18 targets and 31k of 105k candidates.
Running P2 over those survivors measures the actual cascade — filter then match — rather than
a matcher that would never see the unfiltered pool in production. The cost is a recall
ceiling of 16/18 which every P2 number must be read against; `--all` removes the ceiling and
triples the fetch.

Targets are always included regardless of the filter, so a target dropped by P1 still has
text and P2's ceiling stays measurable rather than assumed.

Measured 2026-08-05 on the filtered set: 30,958/31,158 (99%) fetched, 18/18 targets, and
**100% of fetched records carry a non-empty abstract**. An earlier note here warned that
S2 omits many abstracts and that P2 would need a title-only fallback; on this pool that is
not the case, so P2 can assume abstract text.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_hop_pool import OUT_DIR, fetch_metadata, load_meta_cache  # noqa: E402
from sweep_hop_filter import doc_frequency, load_pools, survives  # noqa: E402

# The cut P1 selected on every leave-one-case-out fold.
P1_FWD, P1_BACK, P1_DF = 2, 3, 2


def wanted_ids(pools: dict[str, list[dict]], df: Counter[str], everything: bool) -> list[str]:
    out: set[str] = set()
    for rows in pools.values():
        for row in rows:
            if everything or row["is_target"] or survives(row, df, P1_FWD, P1_BACK, P1_DF):
                out.add(row["id"])
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="fetch the whole pool, not the filtered set")
    args = ap.parse_args()

    pools = load_pools()
    if not pools:
        print("no pools — run build_hop_pool.py first")
        return 1
    df = doc_frequency(pools)
    ids = wanted_ids(pools, df, args.all)
    total = sum(len(r) for r in pools.values())
    print(f"{len(pools)} pools, {total:,} candidates; fetching text for {len(ids):,}")

    cache = load_meta_cache()
    have = sum(1 for i in ids if i in cache)
    print(f"cache already holds {have:,} of them\n")
    cache = fetch_metadata(ids, cache)

    have = sum(1 for i in ids if i in cache)
    tgt = [r["id"] for rows in pools.values() for r in rows if r["is_target"]]
    print(f"\ncoverage: {have:,}/{len(ids):,} ({have / max(len(ids), 1):.0%})")
    print(f"targets with text: {sum(1 for t in tgt if t in cache)}/{len(tgt)}")
    print(f"cache: {OUT_DIR / '_meta_cache.json'}")
    with_abs = sum(1 for i in ids if (cache.get(i) or {}).get("abstract"))
    print(
        f"of those fetched, {with_abs:,} have a non-empty abstract ({with_abs / max(have, 1):.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
