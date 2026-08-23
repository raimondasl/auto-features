"""Where does a new source's contribution die? ($0, judge-free, offline.)

    uv run python evals/source_funnel.py --run <treatment.json> --pool-dir <dir>
    uv run python evals/source_funnel.py --run <treat.json> --pool-dir <dir> \
        --control <control.json>          # adds the displacement table

§20.7 made the funnel the PRIMARY endpoint of a source arm and §21.1 reported one by hand.
This is that table as code, because §38 needs the same one and a table computed twice by hand
is a table computed two different ways.

Four stages, per case, counted by paper origin:

  **pool** — what the collector returned.
  **window** — what survived ranking into `output.top_n`.
  **gated** — what the actionability gate scored >= 2.
  **shown** — what reached Top Picks, which is what a user reads.

Origin is `paper_id.is_arxiv_id`, so a paper OpenAlex or Europe PMC returned that *is* an arXiv
paper counts as arXiv. That is the honest split for a coverage question: a source earns credit
for literature the arXiv channel could not have delivered, not for re-finding it.

**Displacement (§21.4).** Adding a source does not enlarge the digest, it reallocates it. Europe
PMC took 44% of the incumbent Top Picks with it, so "papers gained" and "digest improved" are
different claims and only the paired judged delta speaks to the second. With `--control`, this
counts how many of the control arm's Top Picks survive into the treatment arm's.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id, is_arxiv_id  # noqa: E402

GATE_ACTIONABLE = 2


def _origin(paper_id: str) -> str:
    return "arxiv" if is_arxiv_id(str(paper_id)) else "new"


def funnel(run: Path, pool_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for rec in json.loads(run.read_text(encoding="utf-8")):
        case = rec["case"]
        pool_file = pool_dir / f"{case}.json"
        pool = json.loads(pool_file.read_text(encoding="utf-8"))["candidates"]
        window = rec["returned"]["reporadar_top10"]
        picks = {p["arxiv_id"] for p in rec["returned"]["reporadar_toppicks"]}
        rows.append(
            {
                "case": case,
                "pool": len(pool),
                "pool_new": sum(1 for c in pool if _origin(c["arxiv_id"]) == "new"),
                "window": len(window),
                "window_new": sum(1 for p in window if _origin(p["arxiv_id"]) == "new"),
                "gated_new": sum(
                    1
                    for p in window
                    if _origin(p["arxiv_id"]) == "new"
                    and (p.get("llm_score") or 0) >= GATE_ACTIONABLE
                ),
                "shown": len(picks),
                "shown_new": sum(1 for p in window if p["arxiv_id"] in picks)
                - sum(
                    1
                    for p in window
                    if p["arxiv_id"] in picks and _origin(p["arxiv_id"]) == "arxiv"
                ),
            }
        )
    return rows


def displacement(control: Path, treat: Path) -> list[dict[str, Any]]:
    """How many of the control arm's Top Picks the treatment arm keeps.

    Compared on `dedup_id`, because the same paper can arrive from two sources under two ids
    and counting it as displaced would invent a cost the digest never paid.
    """
    by_case: dict[str, set[str]] = {}
    for rec in json.loads(control.read_text(encoding="utf-8")):
        by_case[rec["case"]] = {
            dedup_id(str(p["arxiv_id"])) for p in rec["returned"]["reporadar_toppicks"]
        }
    rows = []
    for rec in json.loads(treat.read_text(encoding="utf-8")):
        was = by_case.get(rec["case"], set())
        now = {dedup_id(str(p["arxiv_id"])) for p in rec["returned"]["reporadar_toppicks"]}
        rows.append(
            {
                "case": rec["case"],
                "control_picks": len(was),
                "treat_picks": len(now),
                "kept": len(was & now),
                "displaced": len(was - now),
            }
        )
    return rows


DUP_JACCARD = 0.70


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def duplicates(run: Path) -> list[dict[str, Any]]:
    """Papers shown TWICE in one digest under two ids, which `dedup_id` cannot merge.

    An arXiv preprint and its journal version are one paper to a reader and two ids to this
    pipeline: `dedup_id` normalises versions and DOIs, but nothing links `2005.00707` to
    `doi:10.1038/s41524-020-00406-3`. Adding a source that indexes journals therefore admits
    a class of duplicate the arXiv-only pipeline could not produce.

    Matching is token Jaccard on titles at >= 0.70, and **every claimed pair is printed** rather
    than only counted. Title matching is the technique §30 failed on, so this reports its
    evidence instead of asking to be trusted; exact-string matching was tried first and missed
    the CHGNet pair ("CHGNet: Pretrained…" vs "CHGNet as a pretrained…") entirely.
    """
    out = []
    for rec in json.loads(run.read_text(encoding="utf-8")):
        pick_ids = {p["arxiv_id"] for p in rec["returned"]["reporadar_toppicks"]}
        picks = [p for p in rec["returned"]["reporadar_top10"] if p["arxiv_id"] in pick_ids]
        for i, a in enumerate(picks):
            ta = _title_tokens(a.get("title", ""))
            for b in picks[i + 1 :]:
                tb = _title_tokens(b.get("title", ""))
                union = ta | tb
                if not union or len(ta & tb) / len(union) < DUP_JACCARD:
                    continue
                out.append(
                    {
                        "case": rec["case"],
                        "jaccard": len(ta & tb) / len(union),
                        "a": {k: a.get(k) for k in ("arxiv_id", "title", "judge_score")},
                        "b": {k: b.get(k) for k in ("arxiv_id", "title", "judge_score")},
                        "cross_origin": _origin(str(a["arxiv_id"])) != _origin(str(b["arxiv_id"])),
                    }
                )
    return out


def net2_without(run: Path, drop: set[tuple[str, str]]) -> dict[str, float]:
    """Per-case net@2 with a set of (case, id) removed from Top Picks entirely.

    "Removed", not "merged": a merged duplicate would still occupy one slot and score once,
    which is what a fix would produce. Dropping is the cruder accounting and is stated as such
    — it answers "what did these slots contribute", not "what would the fix score".
    """
    out = {}
    for rec in json.loads(run.read_text(encoding="utf-8")):
        pick_ids = {p["arxiv_id"] for p in rec["returned"]["reporadar_toppicks"]}
        kept = [
            p
            for p in rec["returned"]["reporadar_top10"]
            if p["arxiv_id"] in pick_ids and (rec["case"], str(p["arxiv_id"])) not in drop
        ]
        act = sum(1 for p in kept if (p.get("judge_score") or 0) >= GATE_ACTIONABLE)
        out[rec["case"]] = float(act - 2 * (len(kept) - act))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="the TREATMENT artifact")
    ap.add_argument("--pool-dir", required=True, help="the treatment arm's seeded pool directory")
    ap.add_argument("--control", default="", help="control artifact, for the displacement table")
    args = ap.parse_args()

    rows = funnel(Path(args.run), Path(args.pool_dir))
    print("=" * 92)
    print("PRIMARY — the funnel, judge-free: where the new source's papers die")
    print("=" * 92)
    print(
        f"  {'case':18} {'pool':>6} {'of which new':>13} {'window':>7} {'new':>5} "
        f"{'gated>=2':>9} {'shown':>6}"
    )
    for r in rows:
        print(
            f"  {r['case']:18} {r['pool']:6d} {r['pool_new']:13d} {r['window']:7d} "
            f"{r['window_new']:5d} {r['gated_new']:9d} {r['shown_new']:6d}"
        )
    n = len(rows)
    tot = {
        k: sum(r[k] for r in rows)
        for k in ("pool", "pool_new", "window_new", "gated_new", "shown_new")
    }
    print(
        f"  {'TOTAL':18} {tot['pool']:6d} {tot['pool_new']:13d} {'':7} "
        f"{tot['window_new']:5d} {tot['gated_new']:9d} {tot['shown_new']:6d}"
    )
    print(
        f"\n  the new source is {tot['pool_new'] / tot['pool']:.1%} of the merged pool and "
        f"{tot['shown_new'] / n:.2f} papers per case reach Top Picks"
    )

    if args.control:
        drows = displacement(Path(args.control), Path(args.run))
        print("\n" + "=" * 92)
        print("QUATERNARY — displacement: adding a source reallocates the window (§21.4)")
        print("=" * 92)
        print(f"  {'case':18} {'control':>8} {'treat':>6} {'kept':>5} {'displaced':>10}")
        for r in drows:
            print(
                f"  {r['case']:18} {r['control_picks']:8d} {r['treat_picks']:6d} "
                f"{r['kept']:5d} {r['displaced']:10d}"
            )
        was = sum(r["control_picks"] for r in drows)
        gone = sum(r["displaced"] for r in drows)
        print(
            f"  {'TOTAL':18} {was:8d} {sum(r['treat_picks'] for r in drows):6d} "
            f"{sum(r['kept'] for r in drows):5d} {gone:10d}"
        )
        print(f"\n  {gone}/{was} = {gone / was:.1%} of the control's Top Picks displaced")

    dups = duplicates(Path(args.run))
    print("\n" + "=" * 92)
    print(f"THE SAME PAPER, SHOWN TWICE — token-Jaccard >= {DUP_JACCARD}, every pair printed")
    print("=" * 92)
    if not dups:
        print("  none")
    for d in dups:
        tag = "CROSS-ORIGIN" if d["cross_origin"] else "same origin"
        print(f"  {d['case']:16} J={d['jaccard']:.2f}  {tag}")
        for side in ("a", "b"):
            p = d[side]
            print(f"      judge {p['judge_score']}  {str(p['arxiv_id']):40} {p['title'][:48]}")
    if dups:
        cross = [d for d in dups if d["cross_origin"]]
        print(f"\n  {len(dups)} duplicate pairs, {len(cross)} of them cross-origin.")
        # Drop the NEW-source copy: it is the one the treatment added.
        drop = {
            (
                d["case"],
                str(d["a" if _origin(str(d["a"]["arxiv_id"])) == "new" else "b"]["arxiv_id"]),
            )
            for d in cross
        }
        before = net2_without(Path(args.run), set())
        after = net2_without(Path(args.run), drop)
        print(f"  {'case':18} {'as run':>8} {'deduped':>9}")
        for case in before:
            print(f"  {case:18} {before[case]:+8.1f} {after[case]:+9.1f}")
        mb = sum(before.values()) / len(before)
        ma = sum(after.values()) / len(after)
        print(f"  {'MEAN':18} {mb:+8.3f} {ma:+9.3f}     delta {ma - mb:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
