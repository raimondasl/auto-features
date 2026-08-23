"""§23: is the 15-paper window the binding constraint? Analysis of the --rr-window 30 arm.

    uv run python evals/window_arm.py --run <artifact.json> --dry-run   # kill check only, $0
    uv run python evals/window_arm.py --run <artifact.json>             # + second judge
    uv run python evals/window_arm.py --run <artifact.json> --report    # $0 from cache

Every label this project owns stopped at rank 15, because that is what the harness judged. So
"is 15 too small" was a question about papers no judge had seen, and §23 bought 90 of them:
ranks 16-30 of the Europe PMC arm over the six bio cases, from the same frozen pool.

**The kill check runs first and is not optional.** The shipped window is
`rerank_by_actionability(gated)[:15]` and the gate is sampled, so a re-run cannot reproduce it
exactly — but it must land close, or the two runs are different draws and ranks 1-15 cannot serve
as the comparison band. §23.4 set that bar at a mean overlap of **11 of 15**, and it is checked
before any number below is printed.

**Per-paper, deliberately.** The endpoint is 90 papers against 90, not 6 cases against 6. §21.3 is
the standing demonstration of where a per-case endpoint on this exact population lands: a paired
delta of +3.000 with a 95% CI of [-0.385, +6.385], unresolved. Ninety papers a side detects a
difference of roughly 15 points.

**Both judges**, per §19 and §20.8. A paper sitting near a window boundary is a score-2-band paper
by construction, and that is the cell where the two judges agree least — kappa 0.199. An answer
here from one judge would be the same mistake for the fourth time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from label_pool import fisher_exact  # noqa: E402
from run_judge_eval import RESULTS_DIR  # noqa: E402
from second_judge import (  # noqa: E402
    ACTIONABLE,
    DEFAULT_MODEL,
    second_cache_path,
    second_verdict,
    verify_contexts,
)
from second_judge import _load_env as load_env  # noqa: E402

from reporadar.llm_client import LLMError  # noqa: E402
from reporadar.paper_id import is_arxiv_id  # noqa: E402

POOL = EVALS / ".work" / "pool-epmc-treat"
SHIPPED = "judge-gpt-5.5-frozenpool-bigrams_verified-20260821T052215Z.json"
WINDOW = 15  # output.top_n — the band the product actually shows
KILL_OVERLAP = 11.0  # §23.4, mean of 15; the gate is sampled so exact reproduction is not the bar
WIN_GAP = 0.15  # §23.4: ranks 16-30 within 15 points of ranks 1-15 means the cut is arbitrary


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {r["case"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def kill_check(new: dict[str, Any], shipped: dict[str, Any]) -> tuple[float, list[str]]:
    """Mean overlap between the re-run's top-15 and the shipped run's top-15."""
    lines, overlaps = [], []
    for case in sorted(new):
        a = {p["arxiv_id"] for p in new[case]["returned"]["reporadar_top10"][:WINDOW]}
        b = {p["arxiv_id"] for p in shipped[case]["returned"]["reporadar_top10"][:WINDOW]}
        overlaps.append(len(a & b))
        lines.append(f"    {case:16} {len(a & b):2d}/{WINDOW} shared with the shipped window")
    return sum(overlaps) / len(overlaps), lines


def bands(new: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Ranks 1-15 and 16-30, in the rerank order the product would have shown them."""
    out: dict[str, list[dict[str, Any]]] = {"1-15": [], "16-30": []}
    for case in sorted(new):
        ranked = new[case]["returned"]["reporadar_top10"]
        for i, p in enumerate(ranked):
            row = {
                "case": case,
                "rank": i + 1,
                "arxiv_id": p["arxiv_id"],
                "title": p["title"],
                "gate": p.get("llm_score"),
                "gpt_score": p["judge_score"],
                "origin": "arXiv" if is_arxiv_id(p["arxiv_id"]) else "Europe PMC",
            }
            out["1-15" if i < WINDOW else "16-30"].append(row)
    return out


def rate(rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
    return sum(1 for r in rows if r.get(key) is not None and r[key] >= ACTIONABLE), len(rows)


def report_band(label: str, rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
    k, n = rate(rows, key)
    lo, hi = wilson(k, n)
    print(f"    {label:8} {k:3d}/{n:3d} = {k / n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    return k, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--shipped", default=SHIPPED)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    new = load(Path(args.run) if Path(args.run).is_file() else RESULTS_DIR / args.run)
    shipped = load(RESULTS_DIR / args.shipped)

    print("=" * 78)
    print("KILL CHECK — is this the same draw as the run whose window we are questioning?")
    print("=" * 78)
    mean_overlap, lines = kill_check(new, shipped)
    for line in lines:
        print(line)
    print(f"\n    mean overlap {mean_overlap:.1f}/{WINDOW}   bar {KILL_OVERLAP:.0f}/{WINDOW}")
    if mean_overlap < KILL_OVERLAP:
        print(
            "\n    KILL — the re-run is a different draw. Ranks 1-15 here cannot stand in for\n"
            "    the shipped window, so the comparison below would be internal to this run and\n"
            "    §23's question would remain unanswered. Nothing further is reported."
        )
        return 1
    print("    PASS — ranks 1-15 reproduce the shipped window closely enough to compare against.")

    band = bands(new)
    print(f"\n  band sizes: 1-15 n={len(band['1-15'])}   16-30 n={len(band['16-30'])}")
    for name, rows in band.items():
        origins = Counter(r["origin"] for r in rows)
        print(f"    {name:8} origins {dict(origins)}")

    if args.dry_run:
        print("\ndry run — no judge was called.")
        return 0

    # --- second judge over both bands ---
    # Ranks 1-15 carry Sonnet labels from §21's pass, but ONLY for papers that were shown:
    # `second_judge_arm.py` judged Top Picks. Comparing a shown-only 1-15 against a complete
    # 16-30 would compare a selected band with an unselected one and read the selection as a
    # quality cliff. So the unlabelled remainder of 1-15 is judged too; the rest cache-hit.
    already = {
        (r["case"], r["arxiv_id"])
        for r in band["1-15"]
        if second_cache_path(args.model, r["case"], r["arxiv_id"]).is_file()
    }
    targets = band["16-30"] + [r for r in band["1-15"] if (r["case"], r["arxiv_id"]) not in already]
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for r in targets:
        if r["case"] not in by_case:
            pool = json.loads((POOL / f"{r['case']}.json").read_text(encoding="utf-8"))
            by_case[r["case"]] = {c["arxiv_id"]: c for c in pool["candidates"]}
        r["paper"] = by_case[r["case"]].get(r["arxiv_id"])
    missing = [r for r in targets if r["paper"] is None]
    if missing:
        print(
            f"  ! {len(missing)} paper(s) absent from the frozen pool — excluded, never defaulted"
        )
        targets = [r for r in targets if r["paper"] is not None]

    contexts, drifted = verify_contexts(sorted({r["case"] for r in targets}))
    if drifted:
        print(f"  ! {len(drifted)} case(s) EXCLUDED — clone drifted: {drifted}")
        targets = [r for r in targets if r["case"] not in set(drifted)]

    if args.report:
        keep = []
        for r in targets:
            p = second_cache_path(args.model, r["case"], r["arxiv_id"])
            if p.is_file():
                r["sonnet_score"] = int(json.loads(p.read_text(encoding="utf-8"))["score"])
                keep.append(r)
        targets = keep
        print(f"  {len(targets)} second-judge verdicts on disk")
    else:
        load_env()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            futs = {
                pool_exec.submit(
                    second_verdict, r["case"], contexts[r["case"]], r["paper"], args.model
                ): r
                for r in targets
            }
            for fut in as_completed(futs):
                row, done = futs[fut], done + 1
                try:
                    row["sonnet_score"] = fut.result()
                except (LLMError, ValueError, KeyError) as exc:
                    print(f"  ! {row['case']}/{row['arxiv_id']}: {str(exc)[:90]}")
                if done % 20 == 0 or done == len(targets):
                    print(f"  judged {done}/{len(targets)}", flush=True)
        targets = [r for r in targets if "sonnet_score" in r]

    # Ranks 1-15 already have Sonnet labels from §21's second_judge_arm pass, where available.
    for r in band["1-15"]:
        p = second_cache_path(args.model, r["case"], r["arxiv_id"])
        if p.is_file():
            r["sonnet_score"] = int(json.loads(p.read_text(encoding="utf-8"))["score"])

    print("\n" + "=" * 78)
    print("PRIMARY — actionable rate by band, per paper")
    print("=" * 78)
    out: dict[str, Any] = {"mean_overlap": mean_overlap, "judges": {}}
    for judge, name in (("gpt_score", "GPT-5.5"), ("sonnet_score", "Sonnet")):
        # Always the two bands themselves. `targets` is the judging worklist and now carries
        # 1-15 rows as well, so using it as the tail silently mixed the bands — caught by an
        # n of 115 in a band of 90.
        top, tail = band["1-15"], band["16-30"]
        if judge == "sonnet_score":
            top = [r for r in top if r.get("sonnet_score") is not None]
            tail = [r for r in tail if r.get("sonnet_score") is not None]
        print(f"\n  {name}")
        k1, n1 = report_band("1-15", top, judge)
        k2, n2 = report_band("16-30", tail, judge)
        if not n1 or not n2:
            continue
        gap = k1 / n1 - k2 / n2
        p = fisher_exact(k1, n1 - k1, k2, n2 - k2)
        verdict = (
            "WIN — the cut is arbitrary" if gap <= WIN_GAP else "NULL — 15 cuts where quality does"
        )
        print(f"    gap {gap:+.3f}  (WIN bar: <= {WIN_GAP:+.2f})  Fisher p={p:.3f}  ->  {verdict}")
        out["judges"][name] = {
            "top": [k1, n1],
            "tail": [k2, n2],
            "gap": gap,
            "p": p,
            "verdict": verdict,
        }

    verdicts = {v["verdict"].split(" ")[0] for v in out["judges"].values()}
    print("\n" + "=" * 78)
    if len(verdicts) == 1:
        print(f"BOTH JUDGES AGREE: {verdicts.pop()}")
    else:
        print("UNRESOLVED — the judges disagree about which side of the bar this falls on.")
        print("§23.4 named this outcome in advance. Picking the judge that agrees is the")
        print("failure this project keeps catching; it is not available here.")
    print("=" * 78)

    dest = EVALS / ".work" / "window_arm.json"
    out["rows"] = [
        {k: v for k, v in r.items() if k != "paper"} for r in band["1-15"] + band["16-30"]
    ]
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
