"""Is the shipped fine-scale probability map still calibrated on live data?

    uv run python evals/calibrate_finescale.py            # ~$0.30, cached after the first run
    uv run python evals/calibrate_finescale.py --analyse  # re-analyse the cache, $0

`reporadar.finescale` carries two frozen constants, SLOPE and INTERCEPT, fitted offline
against one stored judge run. Everything downstream of them — which papers clear P >= 2/3,
and therefore the +4.55 headline — depends on that map still being located where it was
fitted. Nothing in the test suite can catch it moving: the tests pin the prompt bytes, and
the failure mode is semantic, not textual. This script is the measurement.

**What it measures.** For every paper in RepoRadar's own top-10 across the 22-case
2026-08-09 run, the GPT-5.5 judge already recorded a verdict. That is ground truth for
`actionable == score >= 2`, on exactly the population the map is asked about, with no
selection bias inside the top-10 (the harness judges the whole of it). Re-running the
shipped gate and the shipped map over those same papers gives a prediction to score
against the verdict already on disk.

**The reproduction check comes first and is not optional.** Before any counterfactual is
worth reading, the reconstruction has to arrive at the decision the live run actually
made. If the rebuilt Top Picks set does not match the recorded one, this script is
measuring itself rather than the product, and it says so instead of printing a number.
That is the same discipline `_triage_reporadar` exists to enforce, learned the same way.

**What it cannot do.** It cannot tune the threshold. P >= 2/3 is derived from net@2's own
arithmetic (3p - 2 > 0), not chosen, so "the threshold that scores best here" is not a
result — it is the metric being fitted to itself. The legitimate question is whether the
*map* puts papers on the right side of a fixed line, and the honest counterfactual is a
leave-one-repo-out refit, which never sees the repo it is scored on.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import WORK_DIR, clone_repo, load_benchmark  # noqa: E402
from run_judge_eval import ENV_KEYS, RESULTS_DIR, _triage_reporadar, load_dotenv  # noqa: E402

from reporadar.collector import collect_by_ids  # noqa: E402
from reporadar.config import ProfilerConfig  # noqa: E402
from reporadar.finescale import INTERCEPT, SHOW_THRESHOLD, SLOPE, score_paper  # noqa: E402
from reporadar.llm_client import LLMError  # noqa: E402
from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.profiler import profile_repo  # noqa: E402

EVALS = Path(__file__).resolve().parent
CACHE_DIR = WORK_DIR / "calibration"
DEFAULT_RUN = "judge-gpt-5.5-20260808T234540Z.json"  # the shipped-config (HyDE) arm
ACTIONABLE = 2  # judge score at or above which a paper is worth showing
MIN_ACTIONABLE = 2  # the gate's admit threshold; papers scored exactly this are the band
PROSE_CHARS = 300  # the run's --rr-prose-chars, and the shipped default
TRIAGE_MODEL = "claude-haiku-4-5"
FINESCALE_MODEL = "gpt-4o-mini"


# --------------------------------------------------------------------------- collection


def _cache_path(case: str) -> Path:
    return CACHE_DIR / f"{case}.json"


def _load_cache(case: str) -> dict[str, dict[str, Any]]:
    path = _cache_path(case)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(case: str, rows: dict[str, dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(case).write_text(json.dumps(rows, indent=1, sort_keys=True), encoding="utf-8")


def base_id(arxiv_id: str) -> str:
    """Delegates to the one shared rule; see reporadar.paper_id."""
    return dedup_id(arxiv_id)


def score_case(
    case: str, repo_url: str, records: list[dict[str, Any]], keys: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Gate + fine-scale every top-10 paper of one case. Cached; only successes are kept.

    A failed call is left out of the cache rather than stored as a low score, so a
    transient API error cannot become a permanent data point — the mistake this file's
    sibling experiments made once and paid for by re-running everything.
    """
    cached = _load_cache(case)
    wanted = {base_id(r["arxiv_id"]): r for r in records if r.get("judge_score") is not None}
    missing = [i for i in wanted if i not in cached]
    if not missing:
        print(f"  {case:11} {len(wanted)} papers, all cached")
        return {i: cached[i] for i in wanted}

    print(f"  {case:11} {len(wanted)} papers, {len(missing)} to score")
    papers = collect_by_ids(sorted(missing))
    by_id = {base_id(p["arxiv_id"]): p for p in papers}
    absent = [i for i in missing if i not in by_id]
    if absent:
        # arXiv drops unknown ids silently. Report them; never silently shrink the set.
        print(f"    ! arXiv returned nothing for {len(absent)} id(s): {absent[:5]}")

    dest = clone_repo(repo_url, WORK_DIR / case)
    profile = profile_repo(dest, profiler_cfg=ProfilerConfig(prose_chars=PROSE_CHARS))
    to_score = [by_id[i] for i in missing if i in by_id]

    gate = _triage_reporadar(dest, to_score, keys, TRIAGE_MODEL, PROSE_CHARS)
    fs_cfg = SimpleNamespace(
        openai_api_key=keys.get("OPENAI_API_KEY", ""), openai_model=FINESCALE_MODEL, timeout=60
    )
    n_fs_failed = 0
    for paper in to_score:
        bid = base_id(paper["arxiv_id"])
        row: dict[str, Any] = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper.get("title", ""),
            "judge_score": wanted[bid]["judge_score"],
            "llm_score": gate.get(paper["arxiv_id"], {}).get("llm_score"),
        }
        try:
            expectation, p = score_paper(paper, profile, fs_cfg)
            row["finescale"] = expectation
            row["finescale_p"] = p
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            n_fs_failed += 1
            print(f"    ! fine-scale failed for {bid}: {str(exc)[:100]}")
        cached[bid] = row
    if n_fs_failed:
        print(f"    ! {n_fs_failed} fine-scale failure(s) — omitted, never scored 0")

    _save_cache(case, cached)
    return {i: cached[i] for i in wanted if i in cached}


# ----------------------------------------------------------------------------- analysis


def shown_by_policy(row: dict[str, Any], slope: float, intercept: float) -> bool:
    """The shipped decision: trust the gate above the band, ask the map inside it.

    A band paper with no fine-scale score is NOT shown — "could not score" and "scored
    low" have to stay distinguishable, and the safe reading of the first one is to
    abstain. `reporadar.finescale.enough_scored` makes the same call one level up.
    """
    gate = row.get("llm_score")
    if gate is None or gate < MIN_ACTIONABLE:
        return False
    if gate > MIN_ACTIONABLE:
        return True
    e = row.get("finescale")
    if e is None:
        return False
    return 1.0 / (1.0 + math.exp(-(slope * e + intercept))) >= SHOW_THRESHOLD


def net2(rows: list[dict[str, Any]]) -> float:
    """net@2 over a shown set: +1 per actionable paper, -2 per non-actionable one."""
    return sum(1.0 if r["judge_score"] >= ACTIONABLE else -2.0 for r in rows)


def fit_logistic(
    xs: list[float], ys: list[int], *, steps: int = 4000, lr: float = 0.05
) -> tuple[float, float]:
    """Two-parameter logistic by gradient ascent on the log-likelihood.

    Deliberately the same shape as the frozen map — one slope, one intercept over the
    same expectation — so a refit is comparable to it parameter for parameter, and a
    difference reads as movement rather than as a change of model class.
    """
    slope, intercept = SLOPE, INTERCEPT
    n = len(xs)
    if n == 0:
        return slope, intercept
    for _ in range(steps):
        gs = gi = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, slope * x + intercept))))
            err = y - p
            gs += err * x
            gi += err
        slope += lr * gs / n
        intercept += lr * gi / n
    return slope, intercept


def reliability(rows: list[dict[str, Any]], bins: int = 5) -> list[dict[str, Any]]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        top = b == bins - 1
        sel = [r for r in rows if lo <= r["finescale_p"] < hi or (top and r["finescale_p"] == 1.0)]
        if not sel:
            continue
        out.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(sel),
                "mean_p": sum(r["finescale_p"] for r in sel) / len(sel),
                "empirical": sum(1 for r in sel if r["judge_score"] >= ACTIONABLE) / len(sel),
            }
        )
    return out


def ece(rows: list[dict[str, Any]], bins: int = 5) -> float:
    table = reliability(rows, bins)
    n = len(rows)
    return sum(t["n"] / n * abs(t["mean_p"] - t["empirical"]) for t in table) if n else 0.0


def _rate(rows: list[dict[str, Any]]) -> float | None:
    """Empirical actionable rate, or None on an empty set — never 0.0, which would read
    as "nothing here is actionable" rather than "there is nothing here"."""
    return sum(1 for r in rows if r["judge_score"] >= ACTIONABLE) / len(rows) if rows else None


def _mean_p(rows: list[dict[str, Any]]) -> float | None:
    return sum(r["finescale_p"] for r in rows) / len(rows) if rows else None


def _brier(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum((r["finescale_p"] - float(r["judge_score"] >= ACTIONABLE)) ** 2 for r in rows) / len(
        rows
    )


def auc(rows: list[dict[str, Any]]) -> float | None:
    pos = [r["finescale_p"] for r in rows if r["judge_score"] >= ACTIONABLE]
    neg = [r["finescale_p"] for r in rows if r["judge_score"] < ACTIONABLE]
    if not pos or not neg:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def sign_test(deltas: list[float]) -> tuple[int, int, int, float]:
    """Two-sided exact sign test over per-case deltas; ties are dropped, then reported."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - pos - neg
    n = pos + neg
    if n == 0:
        return pos, neg, ties, 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return pos, neg, ties, min(1.0, 2 * tail)


def bootstrap_ci(
    deltas: list[float], *, iters: int = 10000, seed: int = 20260809
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def analyse(data: dict[str, list[dict[str, Any]]], recorded: dict[str, set[str]]) -> dict[str, Any]:
    cases = sorted(data)
    all_rows = [r for c in cases for r in data[c]]
    scored = [r for r in all_rows if r.get("finescale_p") is not None]
    # Band membership is carried as (case, row) pairs rather than recovered by identity:
    # two repos can legitimately hold the same paper, and a value-equality split would
    # then leak a held-out row into its own training set.
    band_by_case: dict[str, list[dict[str, Any]]] = {
        c: [r for r in data[c] if r.get("llm_score") == MIN_ACTIONABLE and "finescale_p" in r]
        for c in cases
    }
    band = [r for c in cases for r in band_by_case[c]]

    # --- 1. Reproduction. Nothing below is worth reading until this passes. --------
    repro = []
    for c in cases:
        rebuilt = {base_id(r["arxiv_id"]) for r in data[c] if shown_by_policy(r, SLOPE, INTERCEPT)}
        live = recorded.get(c, set())
        repro.append(
            {
                "case": c,
                "rebuilt": len(rebuilt),
                "live": len(live),
                "agree": len(rebuilt & live),
                "only_rebuilt": sorted(rebuilt - live),
                "only_live": sorted(live - rebuilt),
            }
        )
    n_live = sum(r["live"] for r in repro)
    n_agree = sum(r["agree"] for r in repro)

    # --- 2. Calibration of the map on the population it governs -------------------
    per_case = []
    for c in cases:
        cb = band_by_case[c]
        if not cb:
            continue
        emp = sum(1 for r in cb if r["judge_score"] >= ACTIONABLE) / len(cb)
        mp = sum(r["finescale_p"] for r in cb) / len(cb)
        per_case.append(
            {"case": c, "n_band": len(cb), "mean_p": mp, "empirical": emp, "residual": mp - emp}
        )

    # --- 3. LORO refit: the only honest counterfactual -----------------------------
    loro = []
    for c in cases:
        train = [r for other in cases if other != c for r in band_by_case[other]]
        xs = [r["finescale"] for r in train]
        ys = [1 if r["judge_score"] >= ACTIONABLE else 0 for r in train]
        s, i = fit_logistic(xs, ys)
        shipped = net2([r for r in data[c] if shown_by_policy(r, SLOPE, INTERCEPT)])
        refit = net2([r for r in data[c] if shown_by_policy(r, s, i)])
        loro.append(
            {
                "case": c,
                # Recorded so the split is auditable rather than trusted: n_train must
                # equal the total band minus this repo's share, every time. It did not
                # under the first version, which split by value equality and dropped a
                # paper shared between two repositories from both sides.
                "n_train": len(train),
                "slope": s,
                "intercept": i,
                "shipped": shipped,
                "refit": refit,
                "delta": refit - shipped,
            }
        )

    xs_all = [r["finescale"] for r in band]
    ys_all = [1 if r["judge_score"] >= ACTIONABLE else 0 for r in band]
    g_slope, g_intercept = fit_logistic(xs_all, ys_all)

    deltas = [row["delta"] for row in loro]
    pos, neg, ties, p = sign_test(deltas)
    return {
        "run": DEFAULT_RUN,
        "n_papers": len(all_rows),
        "n_scored": len(scored),
        "n_band": len(band),
        "reproduction": {"live": n_live, "agree": n_agree, "cases": repro},
        "calibration": {
            "band": {
                "brier": _brier(band),
                "ece": ece(band),
                "auc": auc(band),
                "reliability": reliability(band),
                "base_rate": _rate(band),
                "mean_p": _mean_p(band),
            },
            "all_top10": {
                "ece": ece(scored),
                "auc": auc(scored),
                "reliability": reliability(scored),
            },
            "per_case": per_case,
        },
        "refit": {
            "frozen": {"slope": SLOPE, "intercept": INTERCEPT},
            "global": {"slope": g_slope, "intercept": g_intercept},
            "loro": loro,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "sign_test": {"pos": pos, "neg": neg, "ties": ties, "p": p},
            "ci95": bootstrap_ci(deltas) if deltas else None,
        },
    }


def report(out: dict[str, Any]) -> None:
    rep = out["reproduction"]
    print("\n" + "=" * 78)
    print("REPRODUCTION — does the rebuilt policy reach the live run's decision?")
    print("=" * 78)
    print(f"  live Top Picks {rep['live']}, rebuilt agrees on {rep['agree']}")
    for r in rep["cases"]:
        if r["only_rebuilt"] or r["only_live"]:
            print(
                f"    {r['case']:11} live {r['live']:2}  rebuilt {r['rebuilt']:2}  "
                f"+{len(r['only_rebuilt'])} / -{len(r['only_live'])}"
            )
    frac = rep["agree"] / rep["live"] if rep["live"] else 0.0
    print(f"  agreement {frac:.0%}")
    if frac < 0.9:
        print("  !! BELOW 90% — the reconstruction is not the product. Read nothing below.")

    cal = out["calibration"]["band"]
    print("\n" + "=" * 78)
    print(f"CALIBRATION on the {out['n_band']} band papers the map actually governs")
    print("=" * 78)
    if cal["base_rate"] is None:
        print("  no band papers — the gate admitted nothing at exactly the threshold")
    else:
        # AUC is undefined when the band is all one class; print that rather than a number.
        auc_txt = f"{cal['auc']:.3f}" if cal["auc"] is not None else "n/a (one class)"
        print(
            f"  base rate {cal['base_rate']:.3f}   mean P {cal['mean_p']:.3f}   "
            f"Brier {cal['brier']:.3f}   ECE {cal['ece']:.3f}   AUC {auc_txt}"
        )
    print(f"\n  {'P bin':10} {'n':>4} {'mean P':>8} {'actual':>8} {'gap':>7}")
    for t in cal["reliability"]:
        print(
            f"  {t['bin']:10} {t['n']:4} {t['mean_p']:8.3f} {t['empirical']:8.3f} "
            f"{t['mean_p'] - t['empirical']:+7.3f}"
        )

    print(f"\n  {'case':11} {'band':>5} {'mean P':>8} {'actual':>8} {'residual':>9}")
    for r in sorted(out["calibration"]["per_case"], key=lambda r: r["residual"]):
        print(
            f"  {r['case']:11} {r['n_band']:5} {r['mean_p']:8.3f} {r['empirical']:8.3f} "
            f"{r['residual']:+9.3f}"
        )

    rf = out["refit"]
    print("\n" + "=" * 78)
    print("LORO REFIT — fit on 21 repos, score the 22nd. Never sees the repo it grades.")
    print("=" * 78)
    print(f"  frozen  slope {rf['frozen']['slope']:.4f}  intercept {rf['frozen']['intercept']:.4f}")
    print(f"  global  slope {rf['global']['slope']:.4f}  intercept {rf['global']['intercept']:.4f}")
    print(f"\n  {'case':11} {'shipped':>8} {'refit':>8} {'delta':>7}")
    for r in sorted(rf["loro"], key=lambda r: r["delta"]):
        print(f"  {r['case']:11} {r['shipped']:8.1f} {r['refit']:8.1f} {r['delta']:+7.1f}")
    st = rf["sign_test"]
    ci = rf["ci95"]
    print(
        f"\n  mean delta {rf['mean_delta']:+.2f} net@2/case   "
        f"sign test {st['pos']}+/{st['neg']}-/{st['ties']}= p = {st['p']:.4f}   "
        f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    )
    if ci[0] <= 0 <= ci[1]:
        print("  The interval spans 0: a refit is NOT shown to beat the frozen map.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=DEFAULT_RUN, help="Results file under evals/results/.")
    ap.add_argument("--analyse", action="store_true", help="Analyse the cache only; no API calls.")
    ap.add_argument("--out", default=str(WORK_DIR / "calibration.json"))
    args = ap.parse_args()

    load_dotenv(EVALS / ".env")
    import os

    keys = {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}
    run = json.loads((RESULTS_DIR / args.run).read_text(encoding="utf-8"))
    bench = {c["name"]: c for c in load_benchmark()["cases"]}

    if not args.analyse:
        for missing in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if missing not in keys:
                raise SystemExit(f"{missing} is required (see evals/README.md); nothing was called")

    data: dict[str, list[dict[str, Any]]] = {}
    recorded: dict[str, set[str]] = {}
    print(f"Scoring the top-10 of {len(run)} cases from {args.run}")
    for rec in run:
        case = rec["case"]
        top10 = rec["returned"]["reporadar_top10"]
        recorded[case] = {base_id(p["arxiv_id"]) for p in rec["returned"]["reporadar_toppicks"]}
        if args.analyse:
            cached = _load_cache(case)
            rows = [cached[b] for p in top10 if (b := base_id(p["arxiv_id"])) in cached]
            print(f"  {case:11} {len(rows)}/{len(top10)} cached")
        else:
            rows = list(score_case(case, bench[case]["live_repo"], top10, keys).values())
        data[case] = rows

    out = analyse(data, recorded)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    report(out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
