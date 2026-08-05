"""P1: can free structural features cut the citation-hop pool without losing the targets?

The hop reaches 18 of 24 known-good papers — the only channel that reaches them at all —
but buries them in 92,014 candidates, 1 good paper per 5,111 (RESEARCH.md §3.5). This sweeps
threshold filters over the persisted pool (`build_hop_pool.py`) and asks one question: is
there a cut that keeps ~all the targets while removing most of the pool?

    uv run python evals/build_hop_pool.py --skip-metadata   # once, ~20 min, free
    uv run python evals/sweep_hop_filter.py                 # instant, offline, free

**Coupling degree is used as a THRESHOLD, never as a sort key.** That is not a stylistic
choice: RETRIEVAL_DESIGN.md's three architectures agree, and it is the one thing they agree
on, that ranking by coupling/similarity/citation-count surfaces the repo's own *ancestry* —
ResNet and COCO for `cv`, BERT and DPR for `rag`. Those are what the repo already knows. A
filter may use the signal to discard; a ranker must not use it to choose.

**Leave-one-case-out.** A threshold tuned on all 7 pools and reported on the same 7 is a
description of those pools. Each case's threshold is chosen on the *other* cases and scored
on the held-out one, so the number generalizes or it does not appear.

Pre-registered before running (ROADMAP P1):
  PREDICTION  some LOO threshold retains >=15/18 targets while cutting the mean seeded-case
              pool by >=75%.
  KILL        if >=5 of 18 targets have fwd_degree <= 1 AND back_degree <= 1, coupling
              cannot separate signal from the noise floor at any threshold — Design 1 dies
              as a corrected-pool negative. (The hop itself survives; its 18/24 is measured
              independently of any filter.)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
POOL_DIR = EVALS / ".work" / "hop_pool"

# The bars pre-registered in ROADMAP P1 were stated against the 18 targets the 12-case
# benchmark could reach. The case set has since grown to 22, so they are held here as
# RATIOS of whatever is actually in the pools — same bar, larger denominator. Restating
# them as ratios is what lets the expanded run be a re-test rather than a new, easier
# question.
RETAIN_FLOOR = 15 / 18  # 83.3% of targets retained
CUT_FLOOR = 0.75  # 75% of the pool removed
KILL_FRACTION = 5 / 18  # 27.8% of targets unreachable from >1 seed either way


def load_pools() -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(POOL_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if rows:
            pools[path.stem] = rows
    return pools


def doc_frequency(pools: dict[str, list[dict[str, Any]]]) -> Counter[str]:
    """How many case pools each candidate appears in.

    A paper reachable from seven unrelated repositories is generic infrastructure —
    Transformer, Adam, BERT — not something specific enough to be actionable for any one of
    them. Free to compute once the pools are persisted, and it needs no metadata.
    """
    df: Counter[str] = Counter()
    for rows in pools.values():
        df.update({r["id"] for r in rows})
    return df


def survives(row: dict[str, Any], df: Counter[str], fwd: int, back: int, df_max: int) -> bool:
    """The filter: keep a candidate if EITHER direction vouches for it, unless it is generic.

    OR rather than AND because the two directions carry different evidence and the backward
    set is small (~2.5% of the pool) but uniquely contributes targets — `cv`'s Soft-NMS is
    backward-only, so an AND (or a forward-only rule) would drop it by construction.
    """
    if df[row["id"]] > df_max:
        return False
    return row.get("fwd_degree", 0) >= fwd or row.get("back_degree", 0) >= back


def evaluate(
    pools: dict[str, list[dict[str, Any]]],
    df: Counter[str],
    fwd: int,
    back: int,
    df_max: int,
    cases: list[str],
) -> tuple[int, int, int]:
    """(targets kept, targets available, candidates kept) over *cases*."""
    kept = avail = size = 0
    for case in cases:
        for row in pools[case]:
            keeps = survives(row, df, fwd, back, df_max)
            if row.get("is_target"):
                avail += 1
                kept += keeps
            size += keeps
    return kept, avail, size


def main() -> int:
    pools = load_pools()
    if not pools:
        print(f"no pools in {POOL_DIR} — run build_hop_pool.py first")
        return 1
    df = doc_frequency(pools)
    n_case = len(pools)
    total_pool = sum(len(r) for r in pools.values())
    total_targets = sum(sum(1 for r in rows if r.get("is_target")) for rows in pools.values())
    print(f"{n_case} pools, {total_pool:,} candidates, {total_targets} targets in pool\n")

    # --- the kill condition, checked before anything else ------------------------------
    lonely = [
        (case, r["id"])
        for case, rows in pools.items()
        for r in rows
        if r.get("is_target") and r.get("fwd_degree", 0) <= 1 and r.get("back_degree", 0) <= 1
    ]
    kill_at = max(1, round(KILL_FRACTION * total_targets))
    print("=== KILL CHECK: targets reachable from <=1 seed in BOTH directions ===")
    print(f"{len(lonely)} of {total_targets}  (kill at >={kill_at}, i.e. {KILL_FRACTION:.0%})")
    for case, tid in lonely:
        print(f"    {case:10} {tid}")
    if len(lonely) >= kill_at:
        print("\n>>> KILL CONDITION MET — coupling cannot separate these from the noise floor.")
    print()

    # --- degree distribution: targets vs the pool they hide in --------------------------
    print("=== degree distribution (target vs non-target) ===")
    print(f"{'':22} {'fwd>=1':>8} {'fwd>=2':>8} {'fwd>=3':>8} {'back>=1':>8} {'back>=2':>8}")
    for label, want in (("targets", True), ("non-targets", False)):
        rows = [r for rr in pools.values() for r in rr if bool(r.get("is_target")) is want]
        n = max(len(rows), 1)
        cells = [sum(1 for r in rows if r.get("fwd_degree", 0) >= k) / n for k in (1, 2, 3)] + [
            sum(1 for r in rows if r.get("back_degree", 0) >= k) / n for k in (1, 2)
        ]
        print(f"{label + f' (n={len(rows)})':22} " + " ".join(f"{c:7.1%}" for c in cells))
    print()

    # --- leave-one-case-out sweep -------------------------------------------------------
    grid = [
        (f, b, d)
        for f in (1, 2, 3, 4)
        for b in (1, 2, 3)
        for d in (n_case, max(n_case - 1, 1), max(n_case // 2, 1), 3, 2)
    ]
    print("=== leave-one-case-out: threshold picked on the others, scored on the held-out ===")
    print(f"{'held out':10} {'chosen (fwd,back,df<=)':>24} {'kept':>10} {'pool cut':>10}")
    loo_kept = loo_avail = 0
    loo_before = loo_after = 0
    for held in sorted(pools):
        others = [c for c in pools if c != held]
        best = None
        for fwd, back, dmax in grid:
            k, a, size = evaluate(pools, df, fwd, back, dmax, others)
            other_total = sum(len(pools[c]) for c in others)
            if a and k / a >= RETAIN_FLOOR and (best is None or size < best[3]):
                best = (fwd, back, dmax, size, other_total)
        if best is None:  # nothing on the grid met the floor on the training cases
            print(f"{held:10} {'(no cut meets the floor)':>24}")
            continue
        fwd, back, dmax, _, _ = best
        k, a, size = evaluate(pools, df, fwd, back, dmax, [held])
        loo_kept += k
        loo_avail += a
        loo_before += len(pools[held])
        loo_after += size
        cut = 1 - size / max(len(pools[held]), 1)
        print(
            f"{held:10} {f'({fwd},{back},{dmax})':>24} {f'{k}/{a}':>10} "
            f"{cut:9.0%}  {len(pools[held]):,} -> {size:,}"
        )

    print()
    print("=== HELD-OUT TOTAL ===")
    if loo_avail:
        cut = 1 - loo_after / max(loo_before, 1)
        print(f"targets retained : {loo_kept}/{loo_avail}")
        print(f"pool             : {loo_before:,} -> {loo_after:,}  ({cut:.0%} cut)")
        print(f"mean per case    : {loo_before // n_case:,} -> {loo_after // n_case:,}")
        need = RETAIN_FLOOR * loo_avail
        ok_r, ok_c = loo_kept >= need, cut >= CUT_FLOOR
        print(
            f"\nPRE-REGISTERED BARS, held as ratios: retain >={RETAIN_FLOOR:.0%} "
            f"({need:.1f} of {loo_avail}) and cut >={CUT_FLOOR:.0%}"
        )
        print(
            f"  retention {loo_kept}/{loo_avail} = {loo_kept / loo_avail:.0%} "
            f"{'MET' if ok_r else 'MISSED'}    cut {cut:.0%} {'MET' if ok_c else 'MISSED'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
