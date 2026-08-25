"""Recompute every published run the comparator's forfeited picks touch [C-25].

Three `cli` baseline caches -- `compiler`, `graph`, `storage` -- hold a 128-character
restoration note where their transcript used to be, after a 30-turn re-run displaced the
12-turn entry on **2026-08-09** and only the identifiers could be recovered. `run_baseline`
re-parses the cached answer on every hit (deliberately, so a parser fix reaches cached
runs), a note parses to nothing, so those three scored `n_returned = 0` -- while
`diagnose_pool.actionable_baseline_ids`, reading the `ids` field of the *same file*, counted
their seven gold targets. One cache, two consumers, opposite answers.

Every 25-case run since that date read the damaged caches, so the understatement is not
confined to the headline. This script finds each affected run and restates it. It re-runs
nothing: net@2 reads a system's own returned papers, so restoring the baseline's picks moves
the baseline column and the paired deltas and leaves RepoRadar's own mean untouched -- the
invariant SS8.10 of the paper already paid to test.

Four rules it follows, each because the alternative has bitten this project:

* **The damage is derived, never hard-coded.** Which caches are damaged comes from
  `baseline._has_answer_block`; what their picks are worth comes from the judge verdicts. A
  cache that is later re-run, or a verdict that later changes, moves this number instead of
  being contradicted by it.
* **Only runs that postdate the damage are restated.** `storage` genuinely abstained in the
  22-case run of 2026-08-07 and `graph` in the 12-case run of 2026-07-12; those are real
  abstentions, and "correcting" them would manufacture picks the run never had.
* **Only `cli` runs are restated.** The run artifacts do **not record which baseline mode
  produced them**, which is its own gap given that [P13] measured `cli` and `api` as
  different systems (64 picks against 34, 10 shared). Mode is therefore recovered by
  matching each run's baseline picks against both caches -- unambiguous in practice: the
  four affected runs match `cli` 45/45, 48/48, 48/48 and 51/51.
* **Both columns are reported.** The published values are what the cited run files contain.
  The correction is that the harness understated the comparator, not that the runs were
  fake, and a restatement that erases the measured value is the C-17 shape wearing a fix's
  clothes.

One artifact of the reproduction, pinned here because it surprised us:
`bigram_report.paired_bootstrap` draws case indices from a **seeded** RNG, so its interval
depends on the ORDER the deltas arrive in and not only on their values. File order gives the
headline [+2.44, +5.96] where sorted order gives the published [+2.44, +6.00] -- one grid
step, decision-irrelevant, and still two people computing "the same" CI from the same run
file and disagreeing. Cases are sorted by name here, which reproduces the published interval.

    uv run python evals/restate_c25.py            # restate every affected run
    uv run python evals/restate_c25.py --check    # $0, diff against the committed artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
from band_testbeds import sign_test  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402
from diagnose_pool import ACTIONABLE, BASELINE, JUDGE, _judge_stem  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
FROZEN = EVALS / "restated_runs.json"

# The 30-turn re-run that displaced the transcripts. A run stamped before this read intact
# caches, so an abstention in it is the baseline's own decision and must be left alone.
DAMAGE_DATE = "20260809"

# The run behind SS8.10's headline table and README's measured-configuration block. Named
# rather than inferred: "the most recent 25-case run" is how a restatement quietly
# re-points itself at a different experiment.
HEADLINE_RUN = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T225831Z.json"
PUBLISHED = {"reporadar": 5.72, "baseline": 1.56, "paired": 4.16}


def _net2(scores: list[int]) -> float:
    return float(sum(1 if s >= ACTIONABLE else -2 for s in scores))


def _run_stamp(name: str) -> str:
    """The `YYYYMMDD` a results filename is stamped with."""
    return name.rsplit("-", 1)[-1][:8]


def forfeited() -> dict[str, list[int]]:
    """``{case: judge scores}`` for every `cli` cache whose `raw` holds no answer block.

    A cache with no recommendation block is not a model that recommended nothing -- an
    explicit ``[]`` is an answer, and `webdev` emits one while still carrying four ids an
    older parser scraped from its prose. Keying on the block is what keeps `webdev` at zero.
    """
    out: dict[str, list[int]] = {}
    for cache in sorted(BASELINE.glob("*.json")):
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("status") != "ok" or baseline_mod._has_answer_block(data.get("raw") or ""):
            continue
        scores = []
        for paper_id in data.get("ids") or []:
            for verdict in (JUDGE / cache.stem).glob(f"{_judge_stem(paper_id)}*.json"):
                scores.append(int(json.loads(verdict.read_text(encoding="utf-8"))["score"]))
                break
        if scores:
            out[cache.stem] = scores
    return out


def _cache_ids(mode: str, case: str) -> set[str] | None:
    path = EVALS / "cache" / "baseline" / mode / f"{case}.json"
    if not path.is_file():
        return None
    return {dedup_id(i) for i in json.loads(path.read_text(encoding="utf-8")).get("ids") or []}


def baseline_mode(run: list[dict[str, Any]]) -> str | None:
    """Which baseline produced this run, recovered from its picks.

    The run artifacts never recorded it. Returns the mode whose caches contain every pick,
    or None when neither does (an older cache state that has since been overwritten).
    """
    best: tuple[float, str] | None = None
    for mode in ("cli", "api"):
        hit = total = 0
        for entry in run:
            ids = {dedup_id(p["arxiv_id"]) for p in (entry["returned"].get("baseline") or [])}
            cached = _cache_ids(mode, entry["case"])
            if cached is None or not ids:
                continue
            hit += len(ids & cached)
            total += len(ids)
        if total and (best is None or hit / total > best[0]):
            best = (hit / total, mode)
    return best[1] if best and best[0] == 1.0 else None


def restate(path: Path, recover: dict[str, list[int]]) -> dict[str, Any] | None:
    """Both columns for one run file, or None if C-25 cannot touch it."""
    run = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(run, list) or not run:
        return None
    if _run_stamp(path.stem) < DAMAGE_DATE or baseline_mode(run) != "cli":
        return None
    # A case whose baseline FAILED carries `net_value@2: None` and was excluded from the
    # published mean -- the +5.42-over-24 row of SS4.2 is exactly that. Restating it over 25
    # would silently re-scope the comparison, so the usable set is recovered rather than
    # assumed, and its size is reported beside the run's.
    usable = [e for e in run if (e["baseline"] or {}).get("net_value@2") is not None]
    if not usable:
        return None

    rows: list[dict[str, Any]] = []
    as_measured: list[float] = []
    corrected: list[float] = []
    for entry in sorted(usable, key=lambda e: e["case"]):
        name = entry["case"]
        rr = float(entry["reporadar_toppicks"]["net_value@2"])
        was = float(entry["baseline"]["net_value@2"])
        now = was
        # Only a case the harness scored as an abstention can have forfeited anything;
        # a cache that replayed its picks is already counted in the published column.
        if name in recover and entry["baseline"]["n_returned"] == 0:
            now = _net2(recover[name])
            rows.append(
                {
                    "case": name,
                    "baseline_was": was,
                    "baseline_now": now,
                    "judge_scores": recover[name],
                }
            )
        as_measured.append(rr - was)
        corrected.append(rr - now)
    if not rows:
        return None

    n = len(usable)
    rr_mean = sum(e["reporadar_toppicks"]["net_value@2"] for e in usable) / n
    shown = sum(e["baseline"]["n_returned"] for e in usable)
    actionable = sum(e["baseline"]["n_actionable"] for e in usable)
    extra_shown = sum(len(r["judge_scores"]) for r in rows)
    extra_good = sum(1 for r in rows for s in r["judge_scores"] if s >= ACTIONABLE)

    def arm(deltas: list[float], *, restored: bool) -> dict[str, Any]:
        lo, hi = paired_bootstrap(deltas)
        st = sign_test(deltas)
        b_shown = shown + (extra_shown if restored else 0)
        b_good = actionable + (extra_good if restored else 0)
        return {
            "reporadar": round(rr_mean, 2),
            "baseline": round(rr_mean - sum(deltas) / n, 2),
            "paired": round(sum(deltas) / n, 2),
            "ci": [round(lo, 2), round(hi, 2)],
            "wins": st["pos"],
            "losses": st["neg"],
            "ties": st["ties"],
            "sign_p": round(st["p"], 6),
            "baseline_shown": b_shown,
            "baseline_actionable": b_good,
            "baseline_precision": round(b_good / b_shown, 3),
        }

    return {
        "run_file": path.name,
        "n_cases": n,
        "n_cases_in_run": len(run),
        "baseline_failed": sorted(e["case"] for e in run if e not in usable),
        "is_headline": path.name == HEADLINE_RUN,
        "forfeited": rows,
        "reporadar_shown": sum(e["reporadar_toppicks"]["n_returned"] for e in usable),
        "reporadar_actionable": sum(e["reporadar_toppicks"]["n_actionable"] for e in usable),
        "as_measured": arm(as_measured, restored=False),
        "corrected": arm(corrected, restored=True),
    }


def build() -> dict[str, Any]:
    recover = forfeited()
    runs = [r for p in sorted(RESULTS.glob("judge-*.json")) if (r := restate(p, recover))]
    headline = next((r for r in runs if r["is_headline"]), None)
    if headline is None:
        raise SystemExit(f"! {HEADLINE_RUN} did not restate; refusing to write a partial artifact.")
    return {
        "_comment": (
            "Every published run C-25 touches, restated. 'as_measured' is what each run file "
            "contains; 'corrected' restores the picks three damaged caches forfeited. "
            "Derived by evals/restate_c25.py; pinned by tests/test_restate_c25.py."
        ),
        "damaged_caches": sorted(recover),
        "damage_date": DAMAGE_DATE,
        "runs": runs,
        # Kept at the top level because the paper, the README and PLANS.md all quote it.
        "run_file": headline["run_file"],
        "n_cases": headline["n_cases"],
        "forfeited": headline["forfeited"],
        "reporadar_shown": headline["reporadar_shown"],
        "reporadar_actionable": headline["reporadar_actionable"],
        "as_measured": headline["as_measured"],
        "corrected": headline["corrected"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Restate every published run C-25 touches.")
    ap.add_argument("--check", action="store_true", help="Diff against the committed artifact.")
    args = ap.parse_args()

    if not (RESULTS / HEADLINE_RUN).is_file():
        print(f"! {HEADLINE_RUN} not present (evals/results/ is gitignored).")
        print("  The committed artifact stands; there is nothing to re-derive here.")
        return 0

    built = build()
    for run in built["runs"]:
        mark = " <- the headline" if run["is_headline"] else ""
        failed = (
            f"  [baseline failed on {', '.join(run['baseline_failed'])}; "
            f"paired over {run['n_cases']} of {run['n_cases_in_run']}]"
            if run["baseline_failed"]
            else ""
        )
        print(f"\n{run['run_file']}{mark}{failed}")
        for label in ("as_measured", "corrected"):
            a = run[label]
            print(
                f"  {label:<12} RepoRadar {a['reporadar']:+.2f}  baseline {a['baseline']:+.2f} "
                f"({a['baseline_shown']}/{a['baseline_actionable']}, "
                f"p={a['baseline_precision']:.3f})"
                f"  paired {a['paired']:+.2f}  CI [{a['ci'][0]:+.2f}, {a['ci'][1]:+.2f}]  "
                f"{a['wins']}w/{a['losses']}l/{a['ties']}t  sign p={a['sign_p']:.4f}"
            )
        print(
            "  forfeited: "
            + ", ".join(
                f"{r['case']} {r['baseline_was']:+.0f}->{r['baseline_now']:+.0f}"
                for r in run["forfeited"]
            )
        )

    measured = built["as_measured"]
    drift = [k for k, v in PUBLISHED.items() if abs(float(measured[k]) - v) > 0.005]
    if drift:
        print(f"\n! the headline's as-measured column no longer reproduces the published {drift}.")
        print("  Fix that before trusting the corrected column -- it shares the same inputs.")
        return 1

    if args.check:
        if not FROZEN.is_file():
            print(f"\n! {FROZEN.name} missing; run without --check to write it.")
            return 1
        stored = json.loads(FROZEN.read_text(encoding="utf-8"))
        if stored.get("runs") != built["runs"]:
            print("\n! the restatement moved; re-run without --check and read the diff.")
            return 1
        print(f"\n{FROZEN.name}: matches.")
        return 0

    FROZEN.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {FROZEN.name} ({len(built['runs'])} affected run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
