"""Second-judge the papers a benchmark arm actually showed, split by where they came from.

    uv run python evals/second_judge_arm.py --run <artifact.json> --dry-run   # $0
    uv run python evals/second_judge_arm.py --run <artifact.json>
    uv run python evals/second_judge_arm.py --run <artifact.json> --report    # $0

§20.8 declared this as part of the Europe PMC arm rather than as a follow-up, and §19 is why:
the second judge reversed the sign of §18.2's headline, and three claims in §17–§19 died on the
one cell where the two judges disagree. Europe PMC abstracts are a distribution neither the gate
nor the fine-scale map was fitted on, so a single-judge precision claim about them would have
been the same mistake a fourth time. Running it in the same pass means the result is either
confirmed or qualified before it is written down, not after someone objects.

**What it answers.** Of the papers an arm put in front of a user, what fraction does an
independent judge call actionable — reported separately for arXiv papers and for the papers the
new source contributed, from the same run, so the comparison is within-arm and needs no
cross-session pairing.

Reuses `second_judge.second_verdict`, so the rubric, model and framing are identical to the
200-label run whose transition table §17.2 quotes. Verdicts cache outside the gold cache.
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

from run_judge_eval import RESULTS_DIR  # noqa: E402
from second_judge import (  # noqa: E402
    ACTIONABLE,
    CACHE,
    DEFAULT_MODEL,
    second_verdict,
    verify_contexts,
)
from second_judge import _load_env as load_env  # noqa: E402

from reporadar.llm_client import LLMError  # noqa: E402
from reporadar.paper_id import is_arxiv_id  # noqa: E402


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval — used because a precision of 1.000 has a Wald CI of zero width,
    which would report perfect certainty from 29 papers."""
    if not n:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def shown_papers(run: list[dict[str, Any]], pool_dir: Path) -> list[dict[str, Any]]:
    """Every paper the arm showed, with the full record from the pool the run used."""
    rows = []
    for rec in run:
        case = rec["case"]
        pool = json.loads((pool_dir / f"{case}.json").read_text(encoding="utf-8"))
        by_id = {c["arxiv_id"]: c for c in pool["candidates"]}
        picks = {p["arxiv_id"] for p in rec["returned"]["reporadar_toppicks"]}
        for p in rec["returned"]["reporadar_top10"]:
            if p["arxiv_id"] not in picks or p["arxiv_id"] not in by_id:
                continue
            rows.append(
                {
                    "case": case,
                    "arxiv_id": p["arxiv_id"],
                    "title": p["title"],
                    "origin": "arXiv" if is_arxiv_id(p["arxiv_id"]) else "new source",
                    "gate": p.get("llm_score"),
                    "gpt_score": p["judge_score"],
                    "paper": by_id[p["arxiv_id"]],
                }
            )
    return rows


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), "groups": {}}
    print("\n" + "=" * 78)
    print("SECOND JUDGE OVER THE SHOWN PAPERS, BY ORIGIN")
    print("=" * 78)
    print(f"  {'origin':12} {'shown':>6} {'GPT prec':>21} {'Sonnet prec':>21}")
    for origin in ("arXiv", "new source"):
        sub = [r for r in rows if r["origin"] == origin]
        if not sub:
            continue
        g = sum(1 for r in sub if r["gpt_score"] >= ACTIONABLE)
        s = sum(1 for r in sub if r["sonnet_score"] >= ACTIONABLE)
        glo, ghi = wilson(g, len(sub))
        slo, shi = wilson(s, len(sub))
        out["groups"][origin] = {
            "n": len(sub),
            "gpt": {"k": g, "precision": g / len(sub), "ci": [glo, ghi]},
            "sonnet": {"k": s, "precision": s / len(sub), "ci": [slo, shi]},
            "gpt_hist": dict(sorted(Counter(r["gpt_score"] for r in sub).items())),
            "sonnet_hist": dict(sorted(Counter(r["sonnet_score"] for r in sub).items())),
        }
        print(
            f"  {origin:12} {len(sub):6d}   {g / len(sub):.3f} [{glo:.3f},{ghi:.3f}]"
            f"     {s / len(sub):.3f} [{slo:.3f},{shi:.3f}]"
        )
    for origin, g in out["groups"].items():
        print(f"\n  {origin}: GPT {g['gpt_hist']}   Sonnet {g['sonnet_hist']}")

    a, b = out["groups"].get("arXiv"), out["groups"].get("new source")
    if a and b:
        print(
            "\n  The comparison that matters is WITHIN this arm and under BOTH judges. A new\n"
            "  source is not established as good because one judge liked it; it is established\n"
            "  as not-worse if it holds its own against the arXiv papers beside it, twice."
        )
        for j in ("gpt", "sonnet"):
            overlap = not (b[j]["ci"][1] < a[j]["ci"][0] or a[j]["ci"][1] < b[j]["ci"][0])
            verdict = "overlapping CIs — not separated" if overlap else "CIs disjoint — separated"
            print(
                f"    {j:6}: arXiv {a[j]['precision']:.3f} vs new source "
                f"{b[j]['precision']:.3f}  -> {verdict}"
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    path = Path(args.run) if Path(args.run).is_file() else RESULTS_DIR / args.run
    run = json.loads(path.read_text(encoding="utf-8"))
    rows = shown_papers(run, Path(args.pool_dir))

    cases = sorted({r["case"] for r in rows})
    contexts, drifted = verify_contexts(cases)
    if drifted:
        print(f"! {len(drifted)} case(s) EXCLUDED — clone drifted under the cache: {drifted}")
        rows = [r for r in rows if r["case"] not in set(drifted)]

    print(f"shown papers: {len(rows)} over {len(contexts)} cases")
    for origin in ("arXiv", "new source"):
        print(f"  {origin:12} {sum(1 for r in rows if r['origin'] == origin):3d}")
    cached = sum(
        1
        for r in rows
        if (CACHE / args.model / r["case"] / f"{r['arxiv_id'].replace('/', '_')}.json").is_file()
    )
    print(f"  cached: {cached}/{len(rows)}   to call: {len(rows) - cached}")
    if args.dry_run:
        print("\ndry run — nothing was called.")
        return 0

    if args.report:
        keep = []
        for r in rows:
            p = CACHE / args.model / r["case"] / f"{r['arxiv_id'].replace('/', '_')}.json"
            if p.is_file():
                r["sonnet_score"] = int(json.loads(p.read_text(encoding="utf-8"))["score"])
                keep.append(r)
        rows = keep
    else:
        load_env()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(
                    second_verdict, r["case"], contexts[r["case"]], r["paper"], args.model
                ): r
                for r in rows
            }
            for fut in as_completed(futs):
                row, done = futs[fut], done + 1
                try:
                    row["sonnet_score"] = fut.result()
                except (LLMError, ValueError, KeyError) as exc:
                    print(f"  ! {row['case']}/{row['arxiv_id']}: {str(exc)[:90]}")
                if done % 20 == 0 or done == len(rows):
                    print(f"  judged {done}/{len(rows)}", flush=True)
        # Never defaulted: a failed call must not become a data point.
        rows = [r for r in rows if "sonnet_score" in r]

    out = report(rows)
    out["rows"] = [{k: v for k, v in r.items() if k != "paper"} for r in rows]
    dest = Path(args.out) if args.out else EVALS / ".work" / "second_judge_arm.json"
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
