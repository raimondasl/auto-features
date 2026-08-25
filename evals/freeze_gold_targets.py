"""Freeze the derived gold set to a committed artifact, with per-id provenance.

The benchmark's gold set is *derived*, not stored: `diagnose_pool.actionable_baseline_ids`
reads the baseline cache and intersects its picks with judge verdicts >= 2. Every published
recall denominator -- 21/56, 34/56, 43/56 -- rests on it. Two facts make that fragile:

* **`evals/cache/` is gitignored.** The 25 `cli` baseline caches exist only on the machine
  that produced them. Nothing in version control holds the gold set.
* **`run_baseline` replays `_parse_recommendations` over the cached `raw` on every hit**
  (baseline.py), so the ids are re-derived rather than read. Re-parsing today yields **51**
  ids against the **64** stored. Nine gold targets exist *only* in the `ids` field:
  `compiler`, `graph` and `storage` carry `raw` = a 128-character restoration note after a
  30-turn re-run displaced the original 12-turn transcript, and `rag` stores two ids its
  own `raw` does not contain. Editing `BASELINE_PROMPT` re-runs those cases and destroys
  them, and the run record they were restored from keeps ids and verdicts but **not** the
  model's answer -- so the reasoning is already gone and cannot be recovered.

This script writes `evals/gold_targets.json`: the derived set, plus for each id whether it
is reproducible from `raw`. `tests/test_gold_targets.py` pins the live derivation against
it, so a change that moves a denominator fails the suite instead of moving silently.

**It freezes 56, not the 51 the parser yields.** Dropping the 9 orphans would change every
published recall figure for a reason unrelated to any research question. They are kept and
*labelled* instead: `provenance: "ids-only"` marks an id the current parser cannot
reproduce, so the weakness is inherited knowingly rather than invisibly.

    uv run python evals/freeze_gold_targets.py           # rewrite the artifact
    uv run python evals/freeze_gold_targets.py --check   # $0, diff against it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
from build_hop_pool import resolve_targets  # noqa: E402
from diagnose_pool import BASELINE  # noqa: E402

EVALS = Path(__file__).resolve().parent
FROZEN = EVALS / "gold_targets.json"


def provenance() -> dict[str, dict[str, str]]:
    """{case: {id: 'raw' | 'ids-only'}} -- can the current parser re-derive this pick?"""
    out: dict[str, dict[str, str]] = {}
    for case, ids in resolve_targets().items():
        if not ids:
            continue
        cache = BASELINE / f"{case}.json"
        reparsed: set[str] = set()
        if cache.is_file():
            data = json.loads(cache.read_text(encoding="utf-8"))
            reparsed = set(baseline_mod._parse_recommendations(data.get("raw") or "")[0])
        out[case] = {i: ("raw" if i in reparsed else "ids-only") for i in ids}
    return out


def build() -> dict[str, object]:
    prov = provenance()
    orphans = {c: [i for i, p in v.items() if p == "ids-only"] for c, v in prov.items()}
    orphans = {c: v for c, v in orphans.items() if v}
    return {
        "_comment": (
            "Frozen gold set. Derived by diagnose_pool.actionable_baseline_ids; pinned by "
            "tests/test_gold_targets.py. 'ids-only' marks a pick the current parser cannot "
            "re-derive from the cached raw answer -- see freeze_gold_targets.py."
        ),
        "n_targets": sum(len(v) for v in prov.values()),
        "n_cases": len(prov),
        "n_ids_only": sum(len(v) for v in orphans.values()),
        "orphans": orphans,
        "targets": {c: sorted(v) for c, v in sorted(prov.items())},
        "provenance": {c: dict(sorted(v.items())) for c, v in sorted(prov.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="freeze/check the gold set")
    ap.add_argument("--check", action="store_true", help="$0: compare, do not write")
    args = ap.parse_args()

    current = build()
    if args.check:
        if not FROZEN.is_file():
            raise SystemExit(f"no frozen artifact at {FROZEN}; run without --check first")
        frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
        same = frozen.get("targets") == current["targets"]
        print(f"frozen : {frozen.get('n_targets')} targets / {frozen.get('n_cases')} cases")
        print(f"derived: {current['n_targets']} targets / {current['n_cases']} cases")
        if same:
            print("MATCH -- the derivation still reproduces the frozen set")
            return 0
        for case in sorted(set(frozen.get("targets", {})) | set(current["targets"])):  # type: ignore[arg-type]
            was = set(frozen.get("targets", {}).get(case, []))
            now = set(current["targets"].get(case, []))  # type: ignore[union-attr]
            if was != now:
                print(f"  {case:12} lost={sorted(was - now)} gained={sorted(now - was)}")
        print("\nDRIFT -- a denominator moved. Intended? Re-freeze deliberately.")
        return 1

    FROZEN.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FROZEN}")
    print(f"  {current['n_targets']} targets across {current['n_cases']} cases")
    print(f"  {current['n_ids_only']} not reproducible from `raw`:")
    for case, ids in sorted(current["orphans"].items()):  # type: ignore[union-attr]
        print(f"    {case:12} {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
