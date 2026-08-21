"""Were the papers the fine-scale stage withheld actually worth showing? (~$3 of Sonnet.)

    uv run python evals/second_judge_band.py --dry-run   # population + hash check, $0
    uv run python evals/second_judge_band.py             # judge the band
    uv run python evals/second_judge_band.py --report    # re-derive from cache, $0

§18.2 measured that the fine-scale stage costs the twelve scientific cases 1.25 net@2 each and
withholds 33 band papers that GPT-5.5 called actionable 82% of the time — against a break-even
of 2/3. That reads as a stage discarding true positives, and it is not usable: **25 of those 27
actionable labels are judge-2s**, the one cell where GPT and Sonnet agree 8 times in 48 (§17.2).
Three claims in §17-§18 died on that cell. This buys the label instead of projecting it.

**Why the shown papers are judged too, when only the withheld ones were asked about.** Sonnet's
base rate on this rubric is **22%**. §18.2's projection puts the withheld set at 19% and 16% —
which is Sonnet's base rate, to within noise. A withheld-only run would therefore produce a
number that cannot distinguish "the map correctly withheld weak papers" from "Sonnet calls
everything weak". The shown band is the control that makes the primary readable, and it is the
§18.4 lesson applied before the fact rather than after: a number without its control is not a
result. It costs about $2 more.

PRE-REGISTERED 2026-08-20, before any call was made.

  POPULATION: all 324 score-2 band papers of the 37-case cohort-3 session (109 scientific,
  215 legacy), split by the shipped map into 80 withheld and 244 shown. Fixed by the artifact,
  not by this script.

  PRIMARY: the Sonnet-actionable rate among the 80 withheld papers, against break-even 2/3.
  CONTROL: the same rate among the 244 shown papers.
  DISCRIMINATION: AUC of `finescale_p` against Sonnet labels, per population.

  BARS, declared now so the number cannot pick one:
    (a) withheld >= 67%  -> the stage discards value under a second judge as well. §18.2's
        projection is refuted and the stage is a real defect, not a metric artefact.
    (b) withheld < 67% AND (shown - withheld) >= 10 points -> withholding is defensible and
        the map discriminates; §18.2's caution was right and the matter is settled.
    (c) withheld < 67% AND (shown - withheld) < 10 points -> the map does not separate these
        papers under this judge. Its withheld rate is just Sonnet's base rate, and NEITHER
        "the stage is correct" NOR "the stage is harmful" is established. This outcome is
        named in advance because it is the likeliest one and the easiest to misreport.

  PREDICTION: (c). §18.2's projection gives 19% and 16%, and Sonnet's base rate is 22%; a
  10-point separation would require the map to be doing something the GPT labels say it does
  not. Recording the prediction because a confirmed (c) is a weak result and it would be easy
  to write it up afterwards as though (b) had been expected.

**Integrity.** Verdicts are cached under `.work/second_judge/`, never in the gold cache, and
this reuses `second_judge.second_verdict` so the rubric and framing are identical to the 200-
label run its transition table comes from. Every case is checked against the stored prompt hash
first: if a clone moved under the cache, the GPT label answers a question we can no longer
rebuild, and the case is excluded rather than noted.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from calibrate_finescale import INTERCEPT, MIN_ACTIONABLE, SLOPE, auc, shown_by_policy  # noqa: E402
from finescale_domains import BREAK_EVEN, DEFAULT_LEGACY, DEFAULT_SCI  # noqa: E402
from second_judge import (  # noqa: E402
    ACTIONABLE,
    CACHE,
    DEFAULT_MODEL,
    WORK,
    cohens_kappa,
    second_verdict,
    verify_contexts,
)
from second_judge import _load_env as load_env  # noqa: E402

from reporadar.llm_client import LLMError  # noqa: E402

POOLS = WORK / "pool-cohort3"
OUT = WORK / "second_judge_band.json"

# Sonnet's base rate on this rubric over the 200-label run (§17.2). The primary endpoint is
# uninterpretable without it: a withheld rate near this number means the map separated nothing.
SONNET_BASE_RATE = 0.22
SEPARATION_BAR = 0.10  # bar (b): shown - withheld, in rate points


def band_papers(path: Path, population: str) -> list[dict[str, Any]]:
    """Every score-2 band paper of one artifact, tagged shown/withheld by the shipped map."""
    rows = []
    for rec in json.loads(path.read_text(encoding="utf-8")):
        for r in rec["returned"]["reporadar_top10"]:
            if r.get("llm_score") != MIN_ACTIONABLE or r.get("finescale_p") is None:
                continue
            rows.append(
                {
                    "case": rec["case"],
                    "population": population,
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "finescale": r["finescale"],
                    "finescale_p": r["finescale_p"],
                    "gpt_score": r["judge_score"],
                    "shown": shown_by_policy(r, SLOPE, INTERCEPT),
                }
            )
    return rows


def resolve_abstracts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach each paper's full record from the frozen pool the run actually used.

    The pool is the run's own input, so this cannot introduce a paper the run never saw, and
    it needs no network. A paper missing from its pool is reported, never skipped silently.
    """
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    resolved, missing = [], []
    for row in rows:
        case = row["case"]
        if case not in by_case:
            pool = json.loads((POOLS / f"{case}.json").read_text(encoding="utf-8"))
            by_case[case] = {c["arxiv_id"]: c for c in pool["candidates"]}
        paper = by_case[case].get(row["arxiv_id"])
        if paper is None or not paper.get("abstract"):
            missing.append(f"{case}/{row['arxiv_id']}")
            continue
        resolved.append({**row, "paper": paper})
    return resolved, missing


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(1 for r in rows if r[key] >= ACTIONABLE) / len(rows) if rows else None


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:5.0%}"


def summarise(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    held = [r for r in rows if not r["shown"]]
    shown = [r for r in rows if r["shown"]]
    r_held, r_shown = rate(held, "sonnet_score"), rate(shown, "sonnet_score")
    sep = None if r_held is None or r_shown is None else r_shown - r_held
    # auc() reads judge_score/finescale_p, so hand it Sonnet's label under that name.
    as_judge = [{"judge_score": r["sonnet_score"], "finescale_p": r["finescale_p"]} for r in rows]
    return {
        "label": label,
        "n": len(rows),
        "n_withheld": len(held),
        "n_shown": len(shown),
        "withheld_rate_gpt": rate(held, "gpt_score"),
        "withheld_rate_sonnet": r_held,
        "shown_rate_gpt": rate(shown, "gpt_score"),
        "shown_rate_sonnet": r_shown,
        "separation": sep,
        "auc_gpt": auc(
            [{"judge_score": r["gpt_score"], "finescale_p": r["finescale_p"]} for r in rows]
        ),
        "auc_sonnet": auc(as_judge),
        "kappa": cohens_kappa(
            [1 if r["gpt_score"] >= ACTIONABLE else 0 for r in rows],
            [1 if r["sonnet_score"] >= ACTIONABLE else 0 for r in rows],
        ),
    }


def verdict_for(summary: dict[str, Any]) -> str:
    """Which pre-registered bar the numbers land on. Named in the docstring, not chosen here."""
    held, sep = summary["withheld_rate_sonnet"], summary["separation"]
    if held is None or sep is None:
        return "no verdict — one arm is empty"
    if held >= BREAK_EVEN:
        return "(a) the stage discards value under a second judge too — §18.2 refuted"
    if sep >= SEPARATION_BAR:
        return "(b) withholding is defensible and the map discriminates"
    return "(c) the map separates nothing here — neither correct nor harmful is established"


def stage_value(rows: list[dict[str, Any]], judge: str, n_cases: int) -> tuple[int, int, float]:
    """net@2 over the band with the fine-scale stage and without it, under one judge's labels.

    "With" is the shown subset; "without" is every band paper, which is §18.3's `--rr-sweep`
    min>=2 arm restricted to the band. Divided by the FULL case count of the population, not
    the count of cases that happen to have band papers, so the figures line up with §18.2's.
    """
    with_stage = sum(1 if r[judge] >= ACTIONABLE else -2 for r in rows if r["shown"])
    without = sum(1 if r[judge] >= ACTIONABLE else -2 for r in rows)
    return with_stage, without, (with_stage - without) / n_cases


def report(rows: list[dict[str, Any]], case_counts: dict[str, int]) -> dict[str, Any]:
    groups = [("ALL-37", rows)] + [
        (p.upper(), [r for r in rows if r["population"] == p]) for p in ("sci", "legacy")
    ]
    out = {"n": len(rows), "groups": []}
    print("\n" + "=" * 78)
    print("SECOND JUDGE OVER THE SCORE-2 BAND")
    print("=" * 78)
    print(f"  {'group':10} {'n':>4} {'withheld':>22} {'shown':>22} {'sep':>6}")
    print(f"  {'':10} {'':4} {'GPT':>10} {'Sonnet':>11} {'GPT':>10} {'Sonnet':>11}")
    for label, sub in groups:
        s = summarise(sub, label)
        out["groups"].append(s)
        sep = "   n/a" if s["separation"] is None else f"{s['separation']:+5.0%}"
        print(
            f"  {label:10} {s['n']:4d} {_fmt(s['withheld_rate_gpt']):>10}"
            f" {_fmt(s['withheld_rate_sonnet']):>11} {_fmt(s['shown_rate_gpt']):>10}"
            f" {_fmt(s['shown_rate_sonnet']):>11} {sep:>6}"
        )
    print(
        f"\n  break-even {BREAK_EVEN:.0%}   Sonnet base rate on this rubric {SONNET_BASE_RATE:.0%}"
        f"   separation bar {SEPARATION_BAR:+.0%}"
    )
    for s in out["groups"]:
        print(f"\n  {s['label']}: {verdict_for(s)}")
        print(
            f"    AUC of finescale_p — against GPT {s['auc_gpt']:.3f}, against Sonnet "
            f"{s['auc_sonnet']:.3f}   (0.5 = the map orders these papers no better than chance)"
        )
        print(f"    kappa between the judges on this band: {s['kappa']:.3f}")

    print("\n" + "=" * 78)
    print("THE STAGE'S VALUE, RECOMPUTED UNDER EACH JUDGE")
    print("=" * 78)
    out["stage_value"] = {}
    for pop, n_cases in case_counts.items():
        sub = [r for r in rows if r["population"] == pop]
        if not sub:
            continue
        line = {}
        for judge, name in (("gpt_score", "GPT-5.5"), ("sonnet_score", "Sonnet")):
            w, wo, per = stage_value(sub, judge, n_cases)
            line[name] = {"with": w, "without": wo, "per_case": per}
            print(
                f"  {pop:7} {name:8} band net@2  with the stage {w:+5d}   without {wo:+5d}"
                f"   the stage is worth {per:+.3f}/case over {n_cases} cases"
            )
        out["stage_value"][pop] = line
    print(
        "\n  Read the SIGN CHANGE, not the levels. Sonnet's base rate on this rubric is "
        f"{SONNET_BASE_RATE:.0%}\n  against GPT's 40%, so every absolute net@2 here is lower "
        "under Sonnet by construction and\n  the negative totals are NOT evidence that the band "
        "should be dropped. What does not\n  depend on the strictness offset is that the two "
        "judges disagree about the stage's\n  sign — which means §18.2's figure was never the "
        "durable half of that section."
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sci", default=DEFAULT_SCI)
    ap.add_argument("--legacy", default=DEFAULT_LEGACY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="population + hash check only, $0")
    ap.add_argument("--report", action="store_true", help="re-derive from cached verdicts, $0")
    ap.add_argument("--workers", type=int, default=8, help="concurrent judge calls.")
    args = ap.parse_args()

    from run_judge_eval import RESULTS_DIR

    rows: list[dict[str, Any]] = []
    case_counts: dict[str, int] = {}
    for name, population in ((args.sci, "sci"), (args.legacy, "legacy")):
        path = Path(name) if Path(name).is_file() else RESULTS_DIR / name
        rows += band_papers(path, population)
        # The FULL case count of the population, including cases with no band paper at all
        # (linter, http and cli have none). §18.2 divided by this, so per-case figures here
        # are directly comparable rather than quietly computed over a smaller denominator.
        case_counts[population] = len(json.loads(path.read_text(encoding="utf-8")))

    rows, missing = resolve_abstracts(rows)
    if missing:
        print(f"! {len(missing)} band paper(s) absent from their frozen pool: {missing[:5]}")

    cases = sorted({r["case"] for r in rows})
    contexts, drifted = verify_contexts(cases)
    if drifted:
        print(f"! {len(drifted)} case(s) EXCLUDED — clone drifted under the cache: {drifted}")
        rows = [r for r in rows if r["case"] not in set(drifted)]

    held = [r for r in rows if not r["shown"]]
    print(
        f"\npopulation: {len(rows)} band papers over {len(contexts)} cases "
        f"({len(held)} withheld, {len(rows) - len(held)} shown)"
    )
    for p in ("sci", "legacy"):
        sub = [r for r in rows if r["population"] == p]
        print(f"  {p:7} {len(sub):4d} band  {sum(1 for r in sub if not r['shown']):3d} withheld")

    cached = sum(
        1
        for r in rows
        if (CACHE / args.model / r["case"] / f"{r['arxiv_id'].replace('/', '_')}.json").is_file()
    )
    print(f"  already cached: {cached}/{len(rows)}   to call: {len(rows) - cached}")
    if args.dry_run:
        print("\ndry run — nothing was called.")
        return 0

    if not args.report:
        load_env()
        # Concurrency is safe here and only changes wall-clock: `llm_client.complete` builds a
        # fresh request per call with no shared client, retries 429/5xx with backoff, and
        # `second_verdict` writes one file per paper. The prompt, model and rubric are
        # untouched, so these verdicts remain comparable with the 200-label run.
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    second_verdict, r["case"], contexts[r["case"]], r["paper"], args.model
                ): r
                for r in rows
            }
            for fut in as_completed(futures):
                row = futures[fut]
                done += 1
                try:
                    row["sonnet_score"] = fut.result()
                except (LLMError, ValueError, KeyError) as exc:
                    # Left unscored rather than defaulted: a failed call must not become a
                    # data point, and a failure that correlates with the verdict would bias
                    # the base rate. second_judge.py learned this from a truncation bug.
                    print(f"  ! {row['case']}/{row['arxiv_id']}: {str(exc)[:90]}")
                if done % 25 == 0 or done == len(rows):
                    print(f"  judged {done}/{len(rows)}", flush=True)
        unscored = [r for r in rows if "sonnet_score" not in r]
        if unscored:
            print(f"  ! {len(unscored)} paper(s) left unscored and EXCLUDED, never defaulted")
            rows = [r for r in rows if "sonnet_score" in r]
    else:
        keep = []
        for r in rows:
            path = CACHE / args.model / r["case"] / f"{r['arxiv_id'].replace('/', '_')}.json"
            if path.is_file():
                r["sonnet_score"] = int(json.loads(path.read_text(encoding="utf-8"))["score"])
                keep.append(r)
        print(f"  {len(keep)}/{len(rows)} verdicts on disk")
        rows = keep

    out = report(rows, case_counts)
    out["rows"] = [{k: v for k, v in r.items() if k != "paper"} for r in rows]
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
