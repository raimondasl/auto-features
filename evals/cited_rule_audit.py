"""§31: does applying the product's already-cited rule move the published headline? ($0.)

    uv run python evals/cited_rule_audit.py

**A validity audit, not a verdict on a rule.** §28.3 found that `evals/run_judge_eval.py` never
applies `profiler.cited_arxiv_ids_of`, while the product does (`cli.py:741`, `profiler.py:939`).
So every published figure here — §16.6's **+5.70 net@2 / 0.894 precision** over 37 cases included
— describes a pipeline that shows papers a real user never sees.

The bar (§31.4) is the project's own paired same-session noise floor, **1.04 net@2/case**: under
it the published numbers stand and the divergence is a footnote; at or over it, §16.6 describes
something that does not ship.

The rule's *merit* is the secondary and is second on purpose. Whether a shipped rule is good is a
smaller question than whether the benchmark has been measuring something else.

**The membership test is the product's own** — `dedup_id(paper_id) in cited_ids`, `digest.py:244`
— not a second implementation of version matching.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from harness import WORK_DIR  # noqa: E402
from run_judge_eval import RESULTS_DIR  # noqa: E402
from second_judge import ACTIONABLE, CACHE, DEFAULT_MODEL  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.profiler import cited_arxiv_ids_of  # noqa: E402

RUNS = (
    ("scientific-12", "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T060917Z.json"),
    ("legacy-25", "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T172033Z.json"),
)
FLOOR = 1.04  # net@2/case, evals/noise_floor.py — §31.4's bar
PUBLISHED_NET = 5.70  # §16.6
PUBLISHED_PREC = 0.894


def net2(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1.0 if r[key] >= ACTIONABLE else -2.0 for r in rows)


def load() -> tuple[list[dict[str, Any]], int]:
    """Every Top Pick of the 37-case session, and the CASE COUNT to divide by.

    The count is every case in the runs, not every case holding a Top Pick. Four cases
    returned nothing at all (§16.4: linter, webdev, http, cli) and an abstention scores a
    legitimate 0 that belongs in the mean. Dividing by 33 instead of 37 inflated the headline
    to +6.394 against §16.6's published +5.70 — caught by this script's own reproduction
    check, which is why it prints the published figure beside the rebuilt one.
    """
    cited: dict[str, frozenset[str]] = {}
    out = []
    n_cases = 0
    for population, fname in RUNS:
        records = json.loads((RESULTS_DIR / fname).read_text(encoding="utf-8"))
        n_cases += len(records)
        for rec in records:
            case = rec["case"]
            if case not in cited:
                cited[case] = cited_arxiv_ids_of(WORK_DIR / case)
            picks = {p["arxiv_id"] for p in rec["returned"]["reporadar_toppicks"]}
            for p in rec["returned"]["reporadar_top10"]:
                if p["arxiv_id"] not in picks:
                    continue
                row = {
                    "case": case,
                    "population": population,
                    "arxiv_id": p["arxiv_id"],
                    "title": p["title"],
                    "judge": p["judge_score"],
                    # The PRODUCT's test (digest.py:244), not a second copy of it.
                    "cited": dedup_id(str(p["arxiv_id"])) in cited[case],
                }
                sp = CACHE / DEFAULT_MODEL / case / f"{p['arxiv_id'].replace('/', '_')}.json"
                if sp.is_file():
                    row["sonnet"] = int(json.loads(sp.read_text(encoding="utf-8"))["score"])
                out.append(row)
    return out, n_cases


def headline(rows: list[dict[str, Any]], n_cases: int, key: str) -> tuple[float, float, int]:
    shown = [r for r in rows if key in r]
    if not shown:
        return (0.0, 0.0, 0)
    act = sum(1 for r in shown if r[key] >= ACTIONABLE)
    return (net2(shown, key) / n_cases, act / len(shown), len(shown))


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows, n_cases = load()
    kept = [r for r in rows if not r["cited"]]
    removed = [r for r in rows if r["cited"]]

    print(f"population: {len(rows)} Top Picks over {n_cases} cases")
    print(f"  the shipped rule would remove: {len(removed)} ({len(removed) / len(rows):.1%})")

    print("\n" + "=" * 92)
    print("PRIMARY — does applying the product's rule move the published headline?")
    print("=" * 92)
    print(f"  {'':22} {'net@2/case':>12} {'precision':>11} {'shown':>7}")
    for label, key in (("GPT-5.5", "judge"), ("Sonnet", "sonnet")):
        before = headline(rows, n_cases, key)
        after = headline(kept, n_cases, key)
        if not before[2]:
            continue
        print(f"  {label} as published    {before[0]:+12.3f} {before[1]:11.3f} {before[2]:7d}")
        print(f"  {label} with the rule   {after[0]:+12.3f} {after[1]:11.3f} {after[2]:7d}")
        d = after[0] - before[0]
        verdict = (
            "under the floor — published numbers stand"
            if abs(d) < FLOOR
            else "AT OR OVER THE FLOOR — §16.6 needs restating"
        )
        print(
            f"  {'':22} delta {d:+.3f}/case, precision {after[1] - before[1]:+.3f}"
            f"   (floor {FLOOR})  ->  {verdict}\n"
        )
    print(f"  §16.6 published +{PUBLISHED_NET} net@2 / {PUBLISHED_PREC} precision; the GPT row")
    print("  above should reproduce it, and any drift is a reconstruction problem, not a finding.")

    print("\n" + "=" * 92)
    print("SECONDARY — what the rule removes, priced in the metric's own arithmetic")
    print("=" * 92)
    for label, key in (("GPT-5.5", "judge"), ("Sonnet", "sonnet")):
        sub = [r for r in removed if key in r]
        if not sub:
            continue
        act = sum(1 for r in sub if r[key] >= ACTIONABLE)
        print(
            f"  {label:9} removes {len(sub):2d} papers: {act} actionable, {len(sub) - act} not"
            f"   -> net@2 {-net2(sub, key):+.1f} over {n_cases} cases"
            f" ({-net2(sub, key) / n_cases:+.3f}/case)"
        )
    by_pop = Counter(r["population"] for r in removed)
    print(f"\n  by domain: {dict(by_pop)}")
    print("  the papers removed:")
    for r in removed:
        s = f" sonnet={r['sonnet']}" if "sonnet" in r else ""
        print(f"    {r['case']:16} judge={r['judge']}{s}  {r['title'][:56]}")

    dest = WORK_DIR / "cited_rule_audit.json"
    dest.write_text(json.dumps({"n": len(rows), "n_removed": len(removed)}, indent=1), "utf-8")
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
