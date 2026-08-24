"""§42 stage 2: does re-anchoring the prose window change what the digest shows? ($0 to report.)

    uv run python evals/prose_anchor_arm.py --control <start.json> --treatment <selfdesc.json>

Both arms score the **same frozen cohort-3 candidates** and differ only in where each repository's
300-character prose window begins (`profiler.prose_anchor`). Prose reaches one consumer,
`triage.py`, so the gate is the only stage that can move — which `evals/prose_window_probe.py`
established by diffing the collector's queries under both values (unchanged 3/3).

**The population is split, and the split is the point (§42.5b).** `bio-align` was chosen *because*
it scores +0.0 net@2, so it is the training example: the rule's shape was fitted to its README. It
cannot also be evidence that the rule generalises. `bio-kmer` and `systems` fell out of the rule
mechanically and were never selected on their scores, so they carry the held-out claim — at n = 2,
which is weak and honest.

**The endpoint is per-paper because three repositories resolve nothing per-repository** (§20.7
learned this the same way). Every paper entering Top Picks contributes +1 if the judge calls it
actionable and −2 if not; every paper leaving contributes the negation of what it used to. Their
sum **is** the net@2 delta, and its n is papers that moved, not repositories.
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

from reporadar.paper_id import dedup_id  # noqa: E402

ACTIONABLE = 2
# §42.5b: chosen because it scores +0.0, so it is the training case and is reported beside the
# held-out pair rather than inside it.
TRAINING_CASE = "bio-align"
WIN_BAR = 3.0  # total net@2 over the held-out cases; ~2x the 1.44 frozen-pool floor for n=3
LOSS_BAR = -3.0
SINGLE_CASE_LOSS = -4.0  # §42.6's guard: a mean must not hide one repository being wrecked


def picks(run: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """case -> dedup id -> the shown paper, with its judge score."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in json.loads(run.read_text(encoding="utf-8")):
        chosen = {p["arxiv_id"] for p in rec["returned"]["reporadar_toppicks"]}
        out[rec["case"]] = {
            dedup_id(str(p["arxiv_id"])): p
            for p in rec["returned"]["reporadar_top10"]
            if p["arxiv_id"] in chosen
        }
    return out


def contribution(paper: dict[str, Any]) -> float:
    """What one shown paper is worth to net@2. Unjudged papers are reported, never scored."""
    score = paper.get("judge_score")
    if score is None:
        return 0.0
    return 1.0 if score >= ACTIONABLE else -2.0


def case_rows(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    added = [pid for pid in after if pid not in before]
    dropped = [pid for pid in before if pid not in after]
    kept = [pid for pid in after if pid in before]
    delta = sum(contribution(after[p]) for p in added) - sum(
        contribution(before[p]) for p in dropped
    )
    return {
        "added": added,
        "dropped": dropped,
        "kept": kept,
        "delta": delta,
        "net_before": sum(contribution(p) for p in before.values()),
        "net_after": sum(contribution(p) for p in after.values()),
        "unjudged": [
            p for p in list(before.values()) + list(after.values()) if p.get("judge_score") is None
        ],
    }


def verdict(held_out_delta: float, worst_case: float) -> str:
    """Which §42.6 bar the numbers land on. Declared there, not chosen here."""
    if worst_case <= SINGLE_CASE_LOSS:
        return (
            f"LOSS — a single case fell {worst_case:+.1f}, past the {SINGLE_CASE_LOSS:+.1f} guard"
        )
    if held_out_delta >= WIN_BAR:
        return "WIN — the held-out cases gain at or above the bar"
    if held_out_delta <= LOSS_BAR:
        return "LOSS — the held-out cases fall at or below the bar"
    return "UNRESOLVED — inside the bars, reported as declared"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment", required=True)
    args = ap.parse_args()

    before, after = picks(Path(args.control)), picks(Path(args.treatment))
    cases = sorted(set(before) & set(after))
    rows = {c: case_rows(before[c], after[c]) for c in cases}

    print("=" * 96)
    print("PER-CASE — Top Picks, and the net@2 each arm earns on them")
    print("=" * 96)
    print(f"  {'case':16} {'shown':>12} {'net@2':>16} {'delta':>8}  {'+/-':>7}")
    for case in cases:
        r = rows[case]
        tag = "  <- TRAINING CASE" if case == TRAINING_CASE else ""
        print(
            f"  {case:16} {len(before[case]):5d} -> {len(after[case]):3d} "
            f"{r['net_before']:+7.1f} -> {r['net_after']:+6.1f} {r['delta']:+8.1f}"
            f"  +{len(r['added'])}/-{len(r['dropped'])}{tag}"
        )

    held = [c for c in cases if c != TRAINING_CASE]
    held_delta = sum(rows[c]["delta"] for c in held)
    worst = min((rows[c]["delta"] for c in cases), default=0.0)

    print("\n" + "=" * 96)
    print(f"PRIMARY — held out ({', '.join(held)}), the cases never chosen on their scores")
    print("=" * 96)
    print(f"  total net@2 delta: {held_delta:+.1f}")
    print(
        f"  bars: WIN >= {WIN_BAR:+.1f}   LOSS <= {LOSS_BAR:+.1f}   "
        f"single-case guard {SINGLE_CASE_LOSS:+.1f}"
    )
    print(f"  VERDICT: {verdict(held_delta, worst)}")

    print("\n" + "=" * 96)
    print(f"SECONDARY — {TRAINING_CASE}, the case the rule was fitted to (n=1, no bar)")
    print("=" * 96)
    if TRAINING_CASE in rows:
        r = rows[TRAINING_CASE]
        print(f"  net@2 {r['net_before']:+.1f} -> {r['net_after']:+.1f}   delta {r['delta']:+.1f}")
        print(
            "  A demonstration that this case is repairable, never evidence the rule generalises."
        )

    print("\n" + "=" * 96)
    print("THE PAPERS THAT MOVED — every one, because n is papers and this is all of them")
    print("=" * 96)
    for case in cases:
        r = rows[case]
        if not r["added"] and not r["dropped"]:
            print(f"  {case}: no membership change")
            continue
        print(f"  {case}:")
        for pid in r["dropped"]:
            p = before[case][pid]
            print(f"    OUT  judge {p.get('judge_score')}  {str(pid):16} {p.get('title', '')[:52]}")
        for pid in r["added"]:
            p = after[case][pid]
            print(f"    IN   judge {p.get('judge_score')}  {str(pid):16} {p.get('title', '')[:52]}")

    print("\n" + "=" * 96)
    print("TERTIARY — precision on what each arm showed")
    print("=" * 96)
    for label, side in (("control", before), ("treatment", after)):
        judged = [p for c in cases for p in side[c].values() if p.get("judge_score") is not None]
        act = sum(1 for p in judged if p["judge_score"] >= ACTIONABLE)
        rate = f"{act}/{len(judged)} = {act / len(judged):.3f}" if judged else "n/a"
        print(f"  {label:10} {rate}")

    stray = [p for c in cases for p in rows[c]["unjudged"]]
    if stray:
        print(f"\n  UNJUDGED papers, scored 0 and reported rather than hidden: {len(stray)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
