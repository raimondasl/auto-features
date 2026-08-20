"""Why did a benchmark case return what it returned? ($0, offline, reads a results artifact.)

    uv run python evals/why_case.py --case linter
    uv run python evals/why_case.py --case linter,webdev,http,cli --run <artifact.json>

`rr why` answers this for a real repository, against the `.reporadar/papers.db` a product run
writes. **The eval harness writes no store**, so the command built for exactly this question
could not be pointed at the data that raised it — the four cases in §16.4 that returned nothing
while holding actionable papers. This is that adapter.

It reports the same stages `reporadar.explain` names, from the fields a results artifact
actually carries (`llm_score`, `judge_score`, tier membership), and says so where the artifact
has no equivalent: the per-paper `score_total`, `rrf_score` and `finescale_p` are not written to
the artifact, so ranking and fine-scale detail are unavailable here in a way they are not in
`rr why`. Nothing is invented to fill a gap.

**Disagreement is reported as disagreement.** The gate (Haiku, 0–3) and the judge (GPT-5.5,
0–3) are two models with different rubrics, and where they differ this script says they differ
— it does not label either one wrong. That is not politeness. `evals/second_judge.py` measured
the judge against Sonnet on 200 labels: Cohen's kappa **0.507** on the ≥2 cut, base rates 40%
vs 22%, and a confusion matrix in which only **8 of 48** papers GPT scored 2 were scored ≥2 by
Sonnet. A gate/judge disagreement sitting on the 1/2 boundary is therefore in the region where
two *judges* disagree most, and calling the gate wrong there asserts something the labelled set
cannot support. Disagreements at the 0 end are a different matter: GPT's 0s were Sonnet's 0s
**58 of 58**.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(EVALS_DIR.parent / "src"))

from metrics import RELEVANT_THRESHOLD  # noqa: E402

RESULTS_DIR = EVALS_DIR / "results"

# Measured in evals/second_judge.py over 200 stratified labels, GPT-5.5 against Sonnet with a
# byte-identical rubric. Quoted here so a reader of this script's output does not have to take
# "disagreement, not error" on trust.
SECOND_JUDGE = {
    0: ("58/58", "100%", "both judges agree at the bottom of the scale"),
    1: ("58/61", "95%", "GPT's 1s stay at or below 1 for Sonnet"),
    2: ("8/48", "17%", "the 1/2 boundary — where the two judges disagree most"),
    3: ("32/33", "97%", "GPT's 3s are Sonnet's 2s or 3s"),
}


def latest_artifact() -> Path:
    runs = sorted(RESULTS_DIR.glob("judge-*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"no results artifacts in {RESULTS_DIR}")
    return runs[-1]


def load(paths: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for p in paths:
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows[r["case"]] = r
    return rows


def report(case: str, row: dict) -> None:
    top10 = row["returned"]["reporadar_top10"]
    picks = {p["arxiv_id"] for p in row["returned"]["reporadar_toppicks"]}
    tp = row["reporadar_toppicks"]

    print(f"\n{'=' * 78}\n{case}  —  {row['repo']}\n{'=' * 78}")
    print(
        f"  pool {row['pool_size']}   actionable in pool {row['n_actionable_in_pool']}   "
        f"Top Picks returned {int(tp['n_returned'])}   net@2 {float(tp['net_value@2']):+.1f}"
    )
    if int(tp["n_returned"]) == 0:
        print("  ABSTAINED — and the metric scores that identically to a correct abstention.")

    gates = collections.Counter(p["llm_score"] for p in top10)
    print(
        f"\n  gate score distribution over the {len(top10)} ranked papers: "
        f"{dict(sorted(gates.items()))}"
    )
    top = gates.most_common(1)[0]
    if top[1] / max(1, len(top10)) >= 0.66:
        print(
            f"    ! {top[1]} of {len(top10)} papers share the single score {top[0]}. A scorer "
            "that puts two thirds of a mixed pool in one bucket is worth looking at on its "
            "own terms, independently of any judge."
        )

    print("\n  where gate and judge disagree (neither is ground truth here):")
    disagree = collections.Counter()
    for p in top10:
        gate_says = p["llm_score"] >= 2  # the shipped min_actionable
        judge_says = p["judge_score"] >= RELEVANT_THRESHOLD
        if gate_says != judge_says:
            disagree[(p["llm_score"], p["judge_score"])] += 1
    if not disagree:
        print("    none — the gate and the judge agree on every ranked paper.")
    for (g, j), n in sorted(disagree.items()):
        frac, pct, note = SECOND_JUDGE.get(j, ("—", "—", ""))
        print(
            f"    gate {g} vs judge {j}: {n:2d} paper(s).  A second judge kept judge-{j} at "
            f"≥2 in {frac} ({pct}) — {note}"
        )

    print("\n  the ranked papers:")
    for p in sorted(top10, key=lambda x: (-x["llm_score"], -x["judge_score"])):
        mark = "TOP" if p["arxiv_id"] in picks else "   "
        flag = (
            "  <-- disagree"
            if (p["llm_score"] >= 2) != (p["judge_score"] >= RELEVANT_THRESHOLD)
            else ""
        )
        print(f"    {mark} gate {p['llm_score']} judge {p['judge_score']}  {p['title'][:58]}{flag}")

    print(
        "\n  Not available from a results artifact: per-paper score_total, rrf_score and "
        "finescale_p.\n  `rr why` reports those against a product store; the eval harness "
        "writes none."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, help="Case name, or a comma-separated list.")
    ap.add_argument("--run", action="append", type=Path, help="Results artifact(s).")
    args = ap.parse_args()

    paths = args.run or [latest_artifact()]
    rows = load(paths)
    wanted = [c.strip() for c in args.case.split(",") if c.strip()]
    missing = [c for c in wanted if c not in rows]
    if missing:
        raise SystemExit(f"not in {[p.name for p in paths]}: {', '.join(missing)}")

    print(f"artifacts: {', '.join(p.name for p in paths)}")
    for case in wanted:
        report(case, rows[case])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
