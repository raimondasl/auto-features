"""Item 5: is the judge's verdict biased by a paper's date relative to its training cutoff?

Re-analysis of stored data only. No LLM calls, no judge calls, no new protocol -- 4,019 cached
GPT-5.5 verdicts and the 837 Claude Sonnet 5 verdicts P7 collected over the same papers.

**The hypothesis (from LitLLMs): the judge has seen pre-cutoff papers in training and rewards
familiarity, so our labels would be systematically kinder to older work.** If true, every
recall and net@2 figure in this project inherits the bias.

**The design does not need to know the cutoff, and deliberately does not guess it.** A
contaminated judge produces a STEP at its own cutoff date. Paper age also predicts
actionability for honest reasons -- a recent paper is likelier to help a current codebase, and
`recency` is one of the ranker's scoring components -- so a smooth trend is expected and is
not the thing being looked for. Testing for a discontinuity anywhere is strictly stronger than
testing at a date we would have had to assume.

**Answer: the hypothesis is not supported, and something larger is in the data instead.**

* The actionability rate does rise with recency, 0.31 (2013) to 0.64 (2025) -- the expected
  trend, no step.
* Then it collapses in the single newest month with volume: **2026-07 scores 0.233 over 159
  papers**, against 0.46-0.67 for every other month of 2026.
* **A second judge sees the same collapse.** Claude Sonnet 5, a different model with a
  different cutoff, reads the same July papers at 0.105 against GPT-5.5's 0.237, and the two
  agree *more* there (0.868) than in any other period. One model's cutoff cannot do that.
* **It is not case mix**: within cases, 10 of 11 fall in July, mean -0.221.
* **It is not the index boundary**: the dense index's newest paper is 2026-07, so July is
  split, but the in-index half also falls (0.333 against June's 0.510).

**What the score distribution says it actually is.** July papers take a *0* -- "no relation to
this repository" -- **52% of the time**, against 0.10-0.13 everywhere else. Unfamiliarity does
not produce that shape; an unfamiliar paper draws a hedged 1 or 2, not a flat rejection. Fifty
per cent outright-unrelated at the highest judged volume of any month is a **retrieval**
symptom: the freshest slice is where off-topic material enters the pool.

**The residual that this data cannot settle, stated rather than buried.** If BOTH judges'
training cutoffs fall in mid-2026, both would be unfamiliar with July papers and both would
mark them down -- two models, one shared blind spot, the same prediction. What is ruled out is
the *single-judge* version of the story. The shared version is not, and the score-0 shape
argues against it without excluding it.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from cited_holdout import _month  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

PRIMARY = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
SECOND = EVALS / ".work" / "second_judge" / "claude-sonnet-5"
INDEX = EVALS / ".work" / "hyde_index"
FROZEN = EVALS / "judge_date_stratify.json"
ACTIONABLE = 2
NEWEST_MONTH = (2026, 7)  # the newest month carrying real volume at collection time


def year_month(paper_id: str) -> tuple[int, int] | None:
    """(year, month) from an arXiv id, or None.

    `_month` returns ``year * 12 + month`` with month in 1..12, so December leaves a
    remainder of 0 and a naive ``m // 12`` reports the FOLLOWING year. Getting that wrong
    moved every December into the next year's bucket in the first draft of this analysis --
    invisible in aggregate, and directly on the boundary the whole probe is about.
    """
    m = _month(paper_id)
    return None if not m else ((m - 1) // 12, (m - 1) % 12 + 1)


def load(root: Path) -> dict[tuple[str, str], int]:
    """(case, base id) -> score, over a cached judge's verdicts."""
    out: dict[tuple[str, str], int] = {}
    for f in root.glob("*/*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "score" in d:
            # dedup_id, never split("v")[0]: the two judges cache under different id forms
            # (`1702.08734v1` against `1702.08734`) and the join is the whole comparison.
            out[(f.parent.name, dedup_id(f.stem))] = int(d["score"])
    return out


def rate(scores) -> float | None:
    scores = list(scores)
    return round(sum(1 for s in scores if s >= ACTIONABLE) / len(scores), 4) if scores else None


def wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 3), round(centre + half, 3)]


def main() -> int:
    gpt, son = load(PRIMARY), load(SECOND)
    paired = sorted(set(gpt) & set(son))
    held = set()
    for f in sorted(INDEX.glob("*.ids")):
        held.update(dedup_id(i) for i in f.read_text(encoding="utf-8").splitlines() if i)

    dated = {k: v for k, v in gpt.items() if year_month(k[1])}
    out: dict[str, Any] = {
        "_comment": (
            "Item 5: judge-contamination re-analysis. Stored verdicts only -- no LLM or judge "
            "calls, no new protocol. Derived by evals/judge_date_stratify.py from "
            "evals/cache/judge/ and evals/.work/second_judge/, both gitignored; pinned by "
            "tests/test_judge_date_stratify.py. VERDICT: the single-judge contamination "
            "hypothesis is NOT supported -- a second model with a different cutoff shows the "
            "same collapse and agrees more, not less. What the data holds instead is a "
            "retrieval symptom in the newest month."
        ),
        "verdicts": {
            "primary_model": "gpt-5.5",
            "primary_total": len(gpt),
            "primary_dated": len(dated),
            "primary_undated": len(gpt) - len(dated),
            "second_model": "claude-sonnet-5",
            "second_total": len(son),
            "paired": len(paired),
        },
        "index_newest_month": list(
            max((year_month(i) for i in held if year_month(i)), default=(0, 0))
        ),
    }

    # -- the trend, and the absence of a step anywhere in it --
    by_year: dict[int, list[int]] = collections.defaultdict(list)
    for (_c, pid), s in dated.items():
        by_year[year_month(pid)[0]].append(s)
    out["by_year"] = {
        str(y): {"n": len(v), "actionable_rate": rate(v)}
        for y, v in sorted(by_year.items())
        if len(v) >= 15
    }

    by_month: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for (_c, pid), s in dated.items():
        d = year_month(pid)
        if d >= (2025, 1):
            by_month[d].append(s)
    out["by_month_recent"] = {
        f"{y}-{m:02d}": {"n": len(v), "actionable_rate": rate(v)}
        for (y, m), v in sorted(by_month.items())
        if len(v) >= 10
    }

    # -- the decisive test: a second model, different cutoff, same papers --
    def slice_paired(pred):
        keys = [k for k in paired if pred(year_month(k[1]))]
        if not keys:
            return None
        return {
            "n": len(keys),
            "gpt_rate": rate(gpt[k] for k in keys),
            "sonnet_rate": rate(son[k] for k in keys),
            "agreement": round(
                sum(1 for k in keys if (gpt[k] >= ACTIONABLE) == (son[k] >= ACTIONABLE))
                / len(keys),
                4,
            ),
        }

    out["two_judges"] = {
        "newest_month": slice_paired(lambda d: d == NEWEST_MONTH),
        "rest_of_2026": slice_paired(lambda d: d and d[0] == 2026 and d[1] < NEWEST_MONTH[1]),
        "2025": slice_paired(lambda d: d and d[0] == 2025),
        "2024_and_earlier": slice_paired(lambda d: d and d[0] <= 2024),
    }

    # -- the shape of the collapse: 0 is "no relation", not "cannot tell" --
    def dist(pred):
        v = [s for (_c, pid), s in dated.items() if pred(year_month(pid))]
        c = collections.Counter(v)
        return {"n": len(v), **{f"score_{k}": round(c[k] / len(v), 3) for k in (0, 1, 2, 3)}}

    out["score_distribution"] = {
        "newest_month": dist(lambda d: d == NEWEST_MONTH),
        "rest_of_2026": dist(lambda d: d and d[0] == 2026 and d[1] < NEWEST_MONTH[1]),
        "2024_2025": dist(lambda d: d and d[0] in (2024, 2025)),
    }
    z = out["score_distribution"]
    out["score_distribution"]["zero_rate_ratio"] = round(
        z["newest_month"]["score_0"] / z["2024_2025"]["score_0"], 2
    )

    # -- the confounds, each checked rather than argued away --
    jul: dict[str, list[int]] = collections.defaultdict(list)
    rest: dict[str, list[int]] = collections.defaultdict(list)
    for (case, pid), s in dated.items():
        d = year_month(pid)
        if d and d[0] == 2026:
            (jul if d[1] == NEWEST_MONTH[1] else rest)[case].append(s)
    rows = [
        {
            "case": c,
            "newest_n": len(jul[c]),
            "newest_rate": rate(jul[c]),
            "other_n": len(rest[c]),
            "other_rate": rate(rest[c]),
            "delta": round(rate(jul[c]) - rate(rest[c]), 3),
        }
        for c in sorted(set(jul) & set(rest))
        if len(jul[c]) >= 5 and len(rest[c]) >= 5
    ]
    out["within_case"] = {
        "_comment": (
            "Case mix ruled out: the collapse happens INSIDE repositories, so it is not an "
            "artefact of which repositories the newest month happens to cover."
        ),
        "cases": rows,
        "n_cases": len(rows),
        "n_falling": sum(1 for r in rows if r["delta"] < 0),
        "mean_delta": round(sum(r["delta"] for r in rows) / len(rows), 3) if rows else None,
    }

    in_idx = [s for (_c, p), s in dated.items() if year_month(p) == NEWEST_MONTH and p in held]
    out_idx = [s for (_c, p), s in dated.items() if year_month(p) == NEWEST_MONTH and p not in held]
    prev = [
        s for (_c, p), s in dated.items() if year_month(p) == (NEWEST_MONTH[0], NEWEST_MONTH[1] - 1)
    ]
    out["index_boundary"] = {
        "_comment": (
            "The dense index's newest paper is in this same month, so the month is split "
            "between what the index holds and what only the live keyword channel could "
            "reach. Channel mix does not explain it either: the IN-index half still falls "
            "well below the previous month."
        ),
        "in_index_n": len(in_idx),
        "in_index_rate": rate(in_idx),
        "not_in_index_n": len(out_idx),
        "not_in_index_rate": rate(out_idx),
        "previous_month_rate": rate(prev),
    }

    n = out["score_distribution"]["newest_month"]["n"]
    k = sum(1 for (_c, p), s in dated.items() if year_month(p) == NEWEST_MONTH and s >= ACTIONABLE)
    out["headline"] = {
        "newest_month": f"{NEWEST_MONTH[0]}-{NEWEST_MONTH[1]:02d}",
        "newest_month_n": n,
        "newest_month_rate": rate(
            [s for (_c, p), s in dated.items() if year_month(p) == NEWEST_MONTH]
        ),
        "newest_month_ci95": wilson(k, n),
        "single_judge_contamination_supported": False,
        "shared_cutoff_excluded": False,
    }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("verdicts", "headline", "two_judges")}, indent=1))
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
