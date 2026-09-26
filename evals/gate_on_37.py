"""The near-binary gate's descriptive figures on all 37 repositories. Post hoc and $0.

Section 5 of the paper describes the gate: how rarely it scores 3, how little the share of 3s says
about a repository's precision, what showing only the 3s costs, and whether the heuristic ranker
orders the score-2 band. Those figures were first measured on the 22-repository development
testbed. This recomputes them on the shipped Haiku gate's run over all 37 repositories (band H of
`finescale_model_transfer.BANDS`), which is also the run behind "19 percent of admitted papers
receive a 3".

That run judged each repository's 15 highest-scored papers with the primary judge, GPT-5.5. The
list is sorted by gate score, and within one score by the heuristic ranker's total
(`anonymous.triage.rerank_by_actionability`), so a paper's position among the score-2 papers is
the ranker's order. In 21 repositories the gate admitted 15 papers or more, and admitted papers
past the fifteenth were never judged. So "admitted" here means admitted within the 15-paper
digest, which is what a reader is shown.

It computes, under the primary judge:

1. the share of admitted papers the gate scores 3 (must reproduce 76 of 404),
2. across repositories with any admitted paper, the Pearson correlation between a repository's
   share of 3s and the precision of its admitted papers, with a repository bootstrap interval,
3. showing only the 3s against showing every admitted paper: mean net@2 per repository over all
   37, precision, and the number of repositories where showing only the 3s shows nothing,
4. the ranker's order within the score-2 band: of the pairs of score-2 papers in one repository
   where the judge accepts one and rejects the other, the share the ranker puts in the judge's
   order (a within-repository AUC), with a repository bootstrap interval.

    uv run python evals/gate_on_37.py      # offline and $0; reads the run file, writes the JSON
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import finescale_model_transfer as fmt  # noqa: E402

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
OUT = EVALS / "gate_on_37.json"

ACTIONABLE = 2
WINDOW = 15
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260926
EXPECTED = {"cases": 37, "admitted": 404, "threes": 76}


def net(papers: list[dict[str, Any]]) -> int:
    """net@2: +1 per actionable paper, -2 per other."""
    return sum(1 if (p.get("judge_score") or 0) >= ACTIONABLE else -2 for p in papers)


def case_rows(run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in run:
        if entry.get("digest_window") != WINDOW:
            raise SystemExit(
                f"{entry['case']}: digest_window {entry.get('digest_window')}, not {WINDOW}"
            )
        listed = entry["returned"]["anonymous_top10"]
        if len(listed) > WINDOW:
            raise SystemExit(f"{entry['case']}: {len(listed)} papers listed, more than {WINDOW}")
        admitted = [p for p in listed if (p.get("llm_score") or 0) >= ACTIONABLE]
        if any(p.get("judge_score") is None for p in admitted):
            raise SystemExit(f"{entry['case']}: an admitted paper has no primary-judge score")
        threes = [p for p in admitted if p["llm_score"] == 3]
        band = [
            (i, (p["judge_score"] or 0) >= ACTIONABLE)
            for i, p in enumerate(listed)
            if p.get("llm_score") == 2
        ]
        pairs = sum(1 for _, a in band if a) * sum(1 for _, a in band if not a)
        ordered = sum(1 for i, a in band if a for k, b in band if not b and i < k)
        rows.append(
            {
                "case": entry["case"],
                "admitted": len(admitted),
                "threes": len(threes),
                "actionable": sum(1 for p in admitted if p["judge_score"] >= ACTIONABLE),
                "actionable_threes": sum(1 for p in threes if p["judge_score"] >= ACTIONABLE),
                "net_all": net(admitted),
                "net_threes": net(threes),
                "band_pairs": pairs,
                "band_pairs_ordered": ordered,
            }
        )
    return rows


def share_precision_r(rows: list[dict[str, Any]]) -> float:
    xs = [r["threes"] / r["admitted"] for r in rows]
    ys = [r["actionable"] / r["admitted"] for r in rows]
    return statistics.correlation(xs, ys)


def ranker_auc(rows: list[dict[str, Any]]) -> float:
    pairs = sum(r["band_pairs"] for r in rows)
    return sum(r["band_pairs_ordered"] for r in rows) / pairs


def bootstrap(rows: list[dict[str, Any]], stat: Any) -> tuple[float, float]:
    """Percentile interval, resampling repositories with the seed fixed. Draws where the statistic
    is undefined (a constant resample, or no informative pair) are skipped and counted."""
    rng = random.Random(BOOTSTRAP_SEED)
    got = []
    for _ in range(BOOTSTRAP_N):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        try:
            got.append(stat(draw))
        except (statistics.StatisticsError, ZeroDivisionError):
            continue
    got.sort()
    return got[int(0.025 * len(got))], got[int(0.975 * len(got)) - 1]


def main() -> int:
    name = fmt.BANDS["H"].run
    run = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    rows = case_rows(run)
    got = {
        "cases": len(rows),
        "admitted": sum(r["admitted"] for r in rows),
        "threes": sum(r["threes"] for r in rows),
    }
    if got != EXPECTED:
        raise SystemExit(f"not the run Section 5 describes: {got}, expected {EXPECTED}")
    nonempty = [r for r in rows if r["admitted"]]
    informative = [r for r in rows if r["band_pairs"]]
    r = share_precision_r(nonempty)
    auc = ranker_auc(informative)
    summary = {
        "run": name,
        "judge": "gpt-5.5",
        "admitted_means": f"admitted within the {WINDOW}-paper digest",
        "share_of_threes": {
            "threes": got["threes"],
            "admitted": got["admitted"],
            "share": round(got["threes"] / got["admitted"], 4),
        },
        "share_vs_precision": {
            "repositories": len(nonempty),
            "pearson_r": round(r, 4),
            "ci95": [round(x, 4) for x in bootstrap(nonempty, share_precision_r)],
        },
        "policies": {
            "repositories": len(rows),
            "show_admitted": {
                "mean_net2": round(statistics.mean(r_["net_all"] for r_ in rows), 4),
                "precision": round(sum(r_["actionable"] for r_ in rows) / got["admitted"], 4),
            },
            "show_only_threes": {
                "mean_net2": round(statistics.mean(r_["net_threes"] for r_ in rows), 4),
                "precision": round(sum(r_["actionable_threes"] for r_ in rows) / got["threes"], 4),
                "repositories_shown_nothing": sum(1 for r_ in rows if not r_["threes"]),
            },
        },
        "ranker_within_band": {
            "repositories": len(informative),
            "pairs": sum(r_["band_pairs"] for r_ in rows),
            "auc": round(auc, 4),
            "ci95": [round(x, 4) for x in bootstrap(informative, ranker_auc)],
        },
    }
    OUT.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT.relative_to(EVALS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
