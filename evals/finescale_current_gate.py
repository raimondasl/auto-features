"""Does the fine-scale rescore still order the gate's modal band?

Registered in `evals/PREREG-finescale-current-gate.md`, committed before any score existed.

The 0.841 band AUC was measured on Testbed A, a 22-case run from 2026-08-07, and every
fine-scale artifact in this project predates 2026-08-09. The gate has moved since: 4.5% to
13.6% of admits scored 3 on the August diagnostics, against 32.1% on the 2026-09-08 sweep. The
modal band is still most of the digest, but it is not the population 0.841 describes.

This scores the current band with the SAME path that produced 0.841: `exp_finescale.score_paper`
unchanged, so the same prompt, model, temperature and logprob reading. A different prompt would
measure a different thing and would not be comparable. The repository side comes from
`band_testbeds.repo_block`, which delegates to the shipped profile builder.

    uv run python evals/finescale_current_gate.py
    uv run python evals/finescale_current_gate.py --limit 2   # smoke, 2 papers per case
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402
import exp_finescale as ef  # noqa: E402

from anonymous.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
RUN = (
    EVALS
    / "results"
    / (
        # The HAIKU control arm of the 2026-09-08 session. The first version of this script used
        # the sibling 163150Z run, which is the Luna arm: ranking_config.rr_gate_model is
        # gpt-5.6-luna there. pool_config.rr_triage_model says claude-haiku-4-5 in BOTH files,
        # because it describes how the pool was collected rather than which gate the run used.
        # Reading that field and stopping is what produced C-37.
        "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T063132Z.json"
    )
)
POOL = EVALS / ".work" / "pool-cut100"
CACHE = EVALS / ".work" / "exp" / "finescale_current_gate_haiku"
OUT = EVALS / "finescale_current_gate.json"
OUT_LUNA = EVALS / "finescale_current_gate_luna.json"

# The 12 scientific cases, for the pre-registered descriptive split.
SCIENTIFIC = {
    "bio-align",
    "bio-kmer",
    "bio-mdsim",
    "bio-mdtraj",
    "bio-scvi",
    "bio-singlecell",
    "mat-chgpot",
    "mat-descriptors",
    "mat-featurize",
    "mat-mlip",
    "mat-phonon",
    "mat-toolkit",
}


def load_band() -> list[tb.Paper]:
    """Every paper the 2026-09-08 sweep gated at exactly 2, with its frozen judge verdict.

    The gate score is read, not reconstructed: unlike Testbed A, this run records
    `llm_score` per shown paper. Abstracts come from the frozen pool the run was scored
    against, so the text is the text the gate saw.
    """
    run = json.loads(RUN.read_text(encoding="utf-8"))
    papers: list[tb.Paper] = []
    for entry in run:
        case = entry["case"]
        pool_file = POOL / f"{case}.json"
        if not pool_file.is_file():
            continue
        abstracts = {
            dedup_id(c.get("arxiv_id") or ""): c
            for c in json.loads(pool_file.read_text(encoding="utf-8"))["candidates"]
        }
        shown = (entry.get("returned") or {}).get("anonymous_top10") or []
        for pos, paper in enumerate(shown):
            if paper.get("llm_score") != 2:
                continue
            record = abstracts.get(dedup_id(paper.get("arxiv_id") or ""))
            if not record or not record.get("abstract"):
                continue
            papers.append(
                tb.Paper(
                    case=case,
                    id=dedup_id(paper.get("arxiv_id") or ""),
                    title=paper.get("title") or "",
                    abstract=record["abstract"],
                    judge=int(paper.get("judge_score") or 0),
                    gate=2,
                    pos=pos,
                )
            )
    return papers


def report(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("exp09") is not None]
    out: dict = {
        "run": RUN.name,
        "model": ef.MODEL,
        "band_papers": len(rows),
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
    }
    if not scored:
        return out

    def auc_of(subset: list[dict]) -> dict:
        labels = [bool(r["judge"] >= tb.ACTIONABLE) for r in subset]
        pos = sum(labels)
        return {
            "n": len(subset),
            "actionable": pos,
            "base_rate": round(pos / len(subset), 4) if subset else None,
            "auc": (
                round(tb.auc([r["exp09"] for r in subset], labels), 4)
                if 0 < pos < len(subset)
                else None
            ),
        }

    # Record WHICH GATE produced this band. C-37: the first version of this script scored a
    # Luna-gated band while believing it was Haiku, because it read pool_config (how the pool
    # was collected) instead of ranking_config (which gate the run used). An artifact that
    # cannot say which gate it measured cannot be checked, so it says.
    run = json.loads(RUN.read_text(encoding="utf-8"))
    rc = run[0].get("ranking_config") or {}
    out["gate"] = {
        "provider": rc.get("rr_gate_provider") or "claude",
        "model": rc.get("rr_gate_model")
        or (run[0].get("pool_config") or {}).get("rr_triage_model"),
    }

    out["overall"] = auc_of(scored)
    out["legacy"] = auc_of([r for r in scored if r["case"] not in SCIENTIFIC])
    out["scientific"] = auc_of([r for r in scored if r["case"] in SCIENTIFIC])
    out["reference"] = {
        "testbed_a_auc": 0.8412228796844181,
        "testbed_a_band_papers": 104,
        "testbed_a_base_rate": 0.75,
        "note": "same scorer and prompt; different run, pool, cohort and date",
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="max band papers per case, for a smoke run")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; source evals/.env first", file=sys.stderr)
        return 2

    papers = load_band()
    if args.limit:
        capped: list[tb.Paper] = []
        seen: dict[str, int] = {}
        for p in papers:
            if seen.get(p.case, 0) >= args.limit:
                continue
            seen[p.case] = seen.get(p.case, 0) + 1
            capped.append(p)
        papers = capped

    print(f"band papers to score: {len(papers)} across {len({p.case for p in papers})} cases")
    CACHE.mkdir(parents=True, exist_ok=True)
    client = ef._client()

    rows: list[dict] = []
    by_case: dict[str, list[tb.Paper]] = {}
    for p in papers:
        by_case.setdefault(p.case, []).append(p)

    for case, group in by_case.items():
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            recs = list(pool.map(lambda p, c=case: ef.score_paper(client, c, p, CACHE), group))
        for paper, rec in zip(group, recs, strict=True):
            rows.append(
                {
                    "case": paper.case,
                    "id": paper.id,
                    "judge": paper.judge,
                    "exp09": rec.get("exp09"),
                    "modal_p": rec.get("modal_p"),
                }
            )
        done = sum(1 for r in rows if r.get("exp09") is not None)
        print(f"  {case:<18} {len(group):>3} papers, {done:>3} scored so far")

    summary = report(rows)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
