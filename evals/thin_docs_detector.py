"""Can the system tell, for free, that it is about to fail on a thin-documentation repo?

    uv run python evals/thin_docs_detector.py

§12.1's finding is not that thin repositories score badly — it is *how* they fail. Queries,
hypotheses, gate and rescore all consume the same impoverished profile and therefore fail
**coherently**: internal consistency is preserved exactly, the gate's top confidence tier
falls to 0.53 while issuing *more* 3s, and the calibrated probability barely moves. "The
judge is the only component seeing the real repository, and thus the only available
detector."

Every remedy tried so far failed, and three of them share a reason:

* a **similarity floor** on the dense channel — refuted, because "the papers are not
  distant, they are the correct answer to the wrong question" (NR-25);
* **stated intent** — +0.12/case with the thinness correlation at the wrong sign (NR-26);
* **source scanning** — −0.52 overall and −2.00 on the thin cohort, worst where aimed
  (NR-36).

The first is the instructive one. A similarity score is computed *from* the corrupted
profile, so it sits inside the coherent-failure loop; so does gate confidence, and so does
the calibrated probability. **Documentation corpus size does not.** It is the number of
characters the profiler read, known *before* a profile exists, deterministic, free, and
structurally incapable of being fooled by a plausible-but-wrong profile.

So this asks a detection question rather than a remedy one, and that matters for what can
be established at this n. A remedy claims "thin repositories score better", which needs a
3-case cohort mean to clear a floor with one case dominating — refuted as unmeasurable.
A detector claims "this signal separates the runs about to degrade from the rest", whose
evidence base is the **24-point ablation grid** (6 repositories × 4 documentation budgets,
where thinness was induced deliberately and the outcome is known at each budget) plus the
**22 rich repositories** for the false-positive rate — the half that decides whether a
detector is usable at all.

**What this cannot show.** In the current configuration almost nothing is net-negative:
abstention scores 0 and there is little left to rescue. So the honest expectation is that
a detector barely moves net@2, and its value is that the user is *told* — which the metric
does not score. That is a real limit, and it is the reason this is cheap rather than the
reason it is valuable.

Corpus size is measured with the profiler's own `_collect_text_corpus`, not a
re-implementation of it: the one-invariant-two-implementations defect is this project's
most repeated correction (C-9, C-12, C-14), and a detector reading a *different* corpus
from the profiler would be measuring a repository nobody profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import EVALS_DIR, WORK_DIR, load_benchmark  # noqa: E402
from metrics import net_actionable_value  # noqa: E402
from run_judge_eval import ablate_docs  # noqa: E402

from reporadar.profiler import _collect_text_corpus  # noqa: E402

RESULTS = EVALS_DIR / "results"

# The four arms of the 2026-08-09 thin-docs ablation grid. The budget is NOT recorded in
# these artifacts — `rr_ablate_docs` was the last POOL_FLAG missing from the recorded
# fields — so each file was identified on 2026-08-16 by matching its mean net@2 against
# the derived summaries in `.work/ablation.json`, exactly (5.1667 / 3.0000 / 3.1667 /
# -0.5000). Written down here so the mapping survives that file, and recorded properly in
# every run from 2026-08-16 onward.
ABLATION_ARMS: dict[str, str] = {
    "control": "judge-gpt-5.5-20260809T052546Z.json",
    "1500": "judge-gpt-5.5-20260809T054713Z.json",
    "300": "judge-gpt-5.5-20260809T061027Z.json",
    "0": "judge-gpt-5.5-20260809T063210Z.json",
}
# A recent run of the shipped configuration, for the false-positive half.
SHIPPED_RUN = "judge-gpt-5.5-bigrams_verified-20260815T041237Z.json"


def corpus_chars(repo_dir: Path) -> int:
    """Characters the profiler actually reads: packaging metadata + README + `docs/`.

    The shipped collector, deliberately. A detector that measured a corpus the profiler
    does not read would be describing a repository nobody profiles.
    """
    return sum(len(doc) for doc in _collect_text_corpus(repo_dir))


def _ranks(values: list[float]) -> list[float]:
    """Average ranks. net@2 is heavily tied (many cases sit at exactly 0.0), and breaking
    those ties by position would manufacture an ordering the data does not contain."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def net_by_case(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for case in d:
        picks = (case.get("returned") or {}).get("reporadar_toppicks")
        if picks is None:
            continue
        scores = [p["judge_score"] for p in picks if p.get("judge_score") is not None]
        out[case["case"]] = net_actionable_value(scores, 2.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Documentation volume as a free failure detector.")
    ap.add_argument(
        "--run",
        default=SHIPPED_RUN,
        help=(
            "results file for the real-repository half. Defaults to the 25-case run NR-37 "
            "used; the frame's P0.4 re-run passes a 37-case one."
        ),
    )
    args = ap.parse_args()
    shipped_run = args.run

    missing = [f for f in [*ABLATION_ARMS.values(), shipped_run] if not (RESULTS / f).exists()]
    if missing:
        print(f"missing run files: {missing}\nnothing measured — a refusal, not a clean result")
        return 1

    arms = {b: net_by_case(RESULTS / f) for b, f in ABLATION_ARMS.items()}
    control = arms["control"]
    repos = sorted(control)

    print("=" * 78)
    print("1. THE GRID — does corpus size track the damage?  (6 repos x 4 budgets)")
    print("=" * 78)
    print(f"\n{'repo':<10} {'budget':>7} {'corpus':>9} {'net@2':>7} {'vs control':>11}")
    grid: list[dict[str, Any]] = []
    for repo in repos:
        for budget in ("control", "1500", "300", "0"):
            src = WORK_DIR / repo
            if not src.exists():
                continue
            path = src if budget == "control" else ablate_docs(src, int(budget))
            chars = corpus_chars(path)
            net = arms[budget].get(repo)
            if net is None:
                continue
            delta = net - control[repo]
            grid.append({"repo": repo, "budget": budget, "chars": chars, "net": net, "d": delta})
            print(f"{repo:<10} {budget:>7} {chars:>9,} {net:>+7.1f} {delta:>+11.1f}")

    degraded = [g for g in grid if g["d"] <= -2.0]
    intact = [g for g in grid if g["d"] > -2.0]
    print(f"\n  points: {len(grid)}   materially degraded (<= -2.0): {len(degraded)}")
    if degraded and intact:
        print(f"  corpus of degraded points: median {st.median(g['chars'] for g in degraded):,.0f}")
        print(f"  corpus of intact points:   median {st.median(g['chars'] for g in intact):,.0f}")

    print("\n" + "=" * 78)
    print("2. THE FALSE-POSITIVE HALF — what would a threshold flag on real repositories?")
    print("=" * 78)
    shipped = net_by_case(RESULTS / shipped_run)
    cases = load_benchmark()["cases"]
    real: list[dict[str, Any]] = []
    for case in cases:
        name = case["name"]
        repo = WORK_DIR / name
        if not repo.exists() or name not in shipped:
            continue
        real.append({"case": name, "chars": corpus_chars(repo), "net": shipped[name]})
    real.sort(key=lambda r: r["chars"])
    print(f"\n{'case':<12} {'corpus':>10} {'net@2':>7}")
    for r in real:
        print(f"{r['case']:<12} {r['chars']:>10,} {r['net']:>+7.1f}")

    # NR-37's substantive finding was the correlation, and the script never computed it —
    # the numbers in RESULTS.md were derived by hand. P0.4 asks for it over 37 cases, so it
    # is emitted here and written to the artifact rather than re-derived a second time.
    usable = [r for r in real if r["chars"] > 0]
    logs = [math.log10(r["chars"]) for r in usable]
    nets = [r["net"] for r in usable]
    pearson, spearman = _pearson(logs, nets), _spearman(logs, nets)
    corr = {
        "run": shipped_run,
        "n": len(usable),
        "pearson_log_corpus_net2": round(pearson, 4),
        "spearman_log_corpus_net2": round(spearman, 4),
        "n_zero_corpus_excluded": len(real) - len(usable),
    }
    print(f"\n  Pearson r(log10 corpus, net@2), n = {corr['n']}: {pearson:+.2f}")
    print(f"  Spearman rho:                          {spearman:+.2f}")
    if corr["n_zero_corpus_excluded"]:
        print(f"  ({corr['n_zero_corpus_excluded']} case(s) with an empty corpus excluded)")

    print("\n" + "=" * 78)
    print("3. THE THRESHOLD SWEEP — what does abstaining below T actually buy?")
    print("=" * 78)
    print("\n  Abstention scores 0, so flagging a case is worth -net@2 when net@2 < 0")
    print("  and costs net@2 when it is positive.\n")
    print(
        f"  {'threshold':>10} {'flagged':>8} {'net@2 as-is':>12} {'if abstained':>13} {'delta':>8}"
    )
    base = st.mean(r["net"] for r in real)
    for t in (0, 500, 1_000, 2_000, 5_000, 10_000, 50_000):
        flagged = [r for r in real if r["chars"] < t]
        after = st.mean(0.0 if r["chars"] < t else r["net"] for r in real)
        print(f"  {t:>10,} {len(flagged):>8} {base:>+12.2f} {after:>+13.2f} {after - base:>+8.2f}")

    out = WORK_DIR / "thin_docs_detector.json"
    out.write_text(
        json.dumps({"grid": grid, "real": real, "correlation": corr}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
