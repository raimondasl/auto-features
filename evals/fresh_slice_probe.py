"""Item 9, and the retraction of the claim that opened it. [C-35]

NR-43 found that the newest month of the judge cache scores 52% "no relation" and concluded:
*the freshest slice is where off-topic material enters the pool, and RepoRadar is a freshness
product whose freshest slice is its worst.* The first half is about the pool and survives. The
second half is about the product and **is false**.

**Two hypotheses were filed with it, and both die here, for free.**

*The stated mechanism does not exist.* NR-43 blamed `recency` weighting -- "a paper from last
month scores ~1.0 on it regardless of topical fit". `w_recency` is **0.0**: in the shipped
default, in `evals/harness.py`, and in every benchmark run since 2026-07-06, because
`--rr-all-time` *is* `w_recency 0`. The ranker does not weight recency at all in the measured
configuration, so it cannot be over-promoting through that channel. Checking the config would
have cost a minute and was not done before the claim was written.

*And the effect is not in the product.* Two measurements, neither needing a new call:

* **Promotion is flat in age.** Over the 37-case arXiv control: pre-2026 papers are shown at
  **146.1 per 10,000 pooled**, 2026 H1 at 172.1, and the newest month at 183.2. The newest
  slice is promoted slightly MORE, on 273 pooled papers and 5 shown -- a difference of about
  one paper, and in the direction that would matter only if those papers were bad. They are
  not: all five are actionable.
* **The newest papers that ARE shown are the best ones shown.** Of 159 judged July papers,
  **11 were ever shown by any run (6.9%, against 36-44% for every other period), and those 11
  are 1.000 actionable.** The 148 never shown score 0.176.

**So the gate is doing its job, visibly and well, exactly where the pool is worst.** The
freshest slice is the hardest slice -- the unshown remainder scores 0.176 against 0.45-0.50
for older years -- and almost none of it reaches a digest.

**The methodological error, which is the part worth keeping.** The judge cache is not a sample
of what the product returns. It is the union of every experiment this project has run:
rank-stratified pool draws (`diagnose_ranker.py` judges ranks 151+ *by design*), off-domain
source arms, ablations, second-judge draws. Stratifying it by date measured **the sampling**,
not the system. The tell was available in the same artifact and not looked at: July's 159
judged papers were the largest month in the cache while being the smallest month in the
digests, and a slice that is simultaneously over-judged and under-shown is a sampling
artifact wearing a finding's clothes.

The contamination result NR-43 was actually written to test is untouched -- that rests on two
judges disagreeing about nothing, not on which papers were sampled.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from cited_holdout import _month  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

RES = EVALS / "results"
POOL = EVALS / ".work" / "pool-core25-arxiv"
JUDGE = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
CONTROL = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json"
FROZEN = EVALS / "fresh_slice_probe.json"
NEWEST = (2026, 7)
ACTIONABLE = 2


def year_month(paper_id: str) -> tuple[int, int] | None:
    m = _month(paper_id)
    return None if not m else ((m - 1) // 12, (m - 1) % 12 + 1)


def rate(scores) -> float | None:
    scores = list(scores)
    return round(sum(1 for s in scores if s >= ACTIONABLE) / len(scores), 4) if scores else None


def main() -> int:
    judged: dict[tuple[str, str], int] = {}
    for f in JUDGE.glob("*/*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "score" in d:
            judged[(f.parent.name, dedup_id(f.stem))] = int(d["score"])

    # Everything the product has ever put in front of a reader, across every run file.
    shown_ever: set[tuple[str, str]] = set()
    n_files = 0
    for f in sorted(RES.glob("judge-*.json")):
        try:
            run = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(run, list):
            continue
        n_files += 1
        for e in run:
            if isinstance(e, dict) and "returned" in e:
                for p in e["returned"].get("reporadar_toppicks") or []:
                    shown_ever.add((e.get("case"), dedup_id(str(p["arxiv_id"]))))

    out: dict[str, Any] = {
        "_comment": (
            "Item 9 / C-35: NR-43's retrieval claim, tested against what the product actually "
            "shows, and retracted. $0 -- stored pools, stored runs, stored verdicts. Derived "
            "by evals/fresh_slice_probe.py; pinned by tests/test_fresh_slice_probe.py. The "
            "judge cache is the union of every experiment ever run, including deliberately "
            "deep pool samples, so stratifying it by date measures the SAMPLING and not the "
            "system. NR-43's contamination result is untouched."
        ),
        "recency_weight_in_measured_config": 0.0,
        "run_files_scanned": n_files,
        "distinct_shown_ever": len(shown_ever),
    }

    # -- 1. promotion rate by age, on the shipped arXiv control --
    pool: dict[str, dict[str, tuple[int, int] | None]] = {}
    for f in sorted(POOL.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        pool[f.stem] = {
            dedup_id(str(c["arxiv_id"])): year_month(str(c["arxiv_id"])) for c in d["candidates"]
        }
    run = json.loads((RES / CONTROL).read_text(encoding="utf-8"))
    shown_here = {
        e["case"]: {
            dedup_id(str(p["arxiv_id"])): int(p["judge_score"])
            for p in e["returned"]["reporadar_toppicks"]
        }
        for e in run
    }
    cases = sorted(set(pool) & set(shown_here))
    pool_by: collections.Counter = collections.Counter()
    shown_by: collections.Counter = collections.Counter()
    for c in cases:
        for _pid, d in pool[c].items():
            if d:
                pool_by[d] += 1
        for pid in shown_here[c]:
            d = pool[c].get(pid) or year_month(pid)
            if d:
                shown_by[d] += 1

    def promo(pred) -> dict[str, Any]:
        p = sum(n for d, n in pool_by.items() if pred(d))
        s = sum(n for d, n in shown_by.items() if pred(d))
        return {
            "pooled": p,
            "shown": s,
            "per_10k": round(s / p * 10000, 1) if p else None,
        }

    out["promotion_by_age"] = {
        "_comment": (
            "If anything promoted fresh papers, they would be shown at a higher rate than "
            "their share of the pool. They are not. Flat is the answer."
        ),
        "run_file": CONTROL,
        "n_cases": len(cases),
        "pool_candidates": sum(len(v) for v in pool.values()),
        "pre_2026": promo(lambda d: d and d < (2026, 1)),
        "2026_h1": promo(lambda d: d and (2026, 1) <= d <= (2026, 6)),
        "newest_month": promo(lambda d: d == NEWEST),
    }

    # -- 2. of what was judged, what was ever shown, and how each half scores --
    periods = {
        "newest_month": lambda d: d == NEWEST,
        "2026_h1": lambda d: d and d[0] == 2026 and d[1] <= 6,
        "2025": lambda d: d and d[0] == 2025,
        "2024_and_older": lambda d: d and d[0] <= 2024,
    }
    out["judged_versus_shown"] = {
        "_comment": (
            "The measurement that retracts the claim. The newest month is the most-judged "
            "month in the cache and the least-shown, and everything shown from it is "
            "actionable. The collapse is in papers the gate declined."
        )
    }
    for label, sel in periods.items():
        keys = [k for k in judged if sel(year_month(k[1]))]
        sh = [k for k in keys if k in shown_ever]
        un = [k for k in keys if k not in shown_ever]
        out["judged_versus_shown"][label] = {
            "judged": len(keys),
            "ever_shown": len(sh),
            "shown_share": round(len(sh) / len(keys), 4) if keys else None,
            "actionable_shown": rate(judged[k] for k in sh),
            "actionable_never_shown": rate(judged[k] for k in un),
        }

    j = out["judged_versus_shown"]
    out["verdict"] = {
        "recency_over_promotion": False,
        "recency_weight_is_zero": True,
        "freshest_slice_defect_in_product": False,
        "gate_declines_the_weak_fresh_papers": True,
        "nr43_retrieval_claim_retracted": True,
        "nr43_contamination_result_unaffected": True,
        "newest_month_shown_share": j["newest_month"]["shown_share"],
        "newest_month_actionable_when_shown": j["newest_month"]["actionable_shown"],
    }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: out[k] for k in ("promotion_by_age", "judged_versus_shown", "verdict")}, indent=1
        )
    )
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
