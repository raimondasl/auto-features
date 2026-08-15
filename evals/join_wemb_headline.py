"""The $0 join: what the headline table reads at `ranking.w_embedding = 1.5`.

Two draws of the 1.5 arm already exist (NR-38), but both ran `--baseline none`, so neither
carries a paired-vs-Opus number. The baseline does not need re-running to supply one:

  * `net_value@2` is a function of a system's OWN returned papers and nothing else
    (`metrics.summarize_system` -> `net_actionable_value(returned_scores, 2.0)`). Pool
    composition drives `ndcg@k` and `pool_has_relevant`; it does not touch net@2.
  * The baseline is an external agent. Its papers are its own, its answers are cached per
    repo, and the judge scores each paper once and caches that too.

So the baseline's per-case net@2 measured on `pool-depth` is the same number it would take
on `pool-wemb`, and joining it against the 1.5 draws is arithmetic, not a new measurement.

**That is an asserted invariant, and this script exists to make it falsifiable.** It prints
a PREDICTION for a real 1.5 arm run with `--baseline cli`. If the paid run disagrees, the
invariant is wrong and the disagreement is worth more than the run cost.

What the join CANNOT produce, and does not print: anything reading `pool_gains` — pool
recall, `n_actionable_in_pool`, ndcg. The draws never had baseline papers in the pool, and
no arithmetic puts them there. Those come from the paid run only.

    uv run python evals/join_wemb_headline.py      # $0, no network, no LLM, no judge
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_report import sign_test  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

# The published headline: 25 cases, window 15, pool-depth, `--baseline cli`, w_embedding at
# the dataclass default. RepoRadar reads +5.12 over 25 cases and +5.42 over the 24 where the
# baseline ran -- the two figures quoted throughout RESULTS.md and the paper.
HEADLINE_RUN = "judge-gpt-5.5-frozenpool-bigrams_verified-20260814T175835Z.json"
HEADLINE_MEAN_25 = 5.12  # asserted, so pointing this at the wrong artifact is caught

# The four pool-wemb arms of NR-38. Same pool, same window, same session pair per draw.
DRAWS: dict[str, tuple[str, float]] = {
    "control draw 1": ("judge-gpt-5.5-frozenpool-bigrams_verified-20260815T161827Z.json", 0.0),
    "control draw 2": ("judge-gpt-5.5-frozenpool-bigrams_verified-20260815T180449Z.json", 0.0),
    "treat   draw 1": (
        "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T170226Z.json",
        1.5,
    ),
    "treat   draw 2": (
        "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T184616Z.json",
        1.5,
    ),
}

# The arm the prediction was written for: same command as draw 2's treatment except
# `--baseline cli` replaces `--baseline none`. One variable.
PAID_ARM = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T225831Z.json"

# The two-draw averaged floor. A single draw carries the window-15 frozen floor of 0.74.
FLOOR_TWO_DRAW = 0.52
FLOOR_ONE_DRAW = 0.74

Run = dict[str, dict[str, Any]]


def load(name: str) -> Run:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing artifact {path}\nThe join cannot substitute for a run.")
    return {r["case"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def check_draw(label: str, run: Run, expected_w: float) -> None:
    """A mislabelled file is a silent arm swap; refuse rather than average the wrong thing."""
    got = {r.get("w_embedding") for r in run.values()}
    if got != {expected_w}:
        raise SystemExit(f"{label}: recorded w_embedding {got}, expected {{{expected_w}}}")
    windows = {r.get("digest_window") for r in run.values()}
    if windows != {15}:
        raise SystemExit(f"{label}: digest_window {windows}, expected all 15")
    if {r.get("baseline_status") for r in run.values()} != {"skipped"}:
        raise SystemExit(f"{label}: expected a --baseline none arm; this one ran a baseline")


def check_same_pool(runs: dict[str, Run]) -> None:
    """Every arm must sit on the identical per-case frozen pool, or the pairing is fiction."""
    labels = list(runs)
    cases = set(runs[labels[0]])
    for label in labels[1:]:
        if set(runs[label]) != cases:
            raise SystemExit(f"{label}: case set differs from {labels[0]}")
    bad = [
        case
        for case in sorted(cases)
        if len({runs[x][case]["pool_provenance"]["fingerprint"] for x in labels}) != 1
    ]
    if bad:
        raise SystemExit(f"arms sit on different pools for {len(bad)} case(s): {bad}")


def check_headline(run: Run) -> None:
    got = {r.get("w_embedding") for r in run.values()}
    # `None` predates the recorded field; the flag did not exist, so the run took the
    # dataclass default of 0.0. Any other value means this is not the published headline.
    if not got <= {None, 0.0}:
        raise SystemExit(f"headline artifact records w_embedding {got}, expected 0.0/None")
    mean = statistics.mean(net(run).values())
    if abs(mean - HEADLINE_MEAN_25) > 0.005:
        raise SystemExit(
            f"headline artifact means {mean:+.2f} over 25 cases, expected "
            f"{HEADLINE_MEAN_25:+.2f} -- wrong file, or the metric moved under it"
        )


def net(run: Run, key: str = "reporadar_toppicks") -> dict[str, float]:
    return {c: float(r[key]["net_value@2"]) for c, r in run.items()}


def baseline_net(run: Run) -> tuple[dict[str, float], list[str]]:
    """Per-case baseline net@2, and the cases where the baseline did NOT run.

    A failed baseline emits no metric numbers by design, so those cases are excluded and
    NAMED -- never read as a legitimate 0.0.
    """
    ok = {
        c: float(r["baseline"]["net_value@2"])
        for c, r in run.items()
        if r["baseline_status"] == "ok"
    }
    failed = sorted(c for c, r in run.items() if r["baseline_status"] != "ok")
    return ok, failed


def volume(run: Run, cases: list[str], key: str = "reporadar_toppicks") -> tuple[int, int, float]:
    shown = sum(int(run[c][key]["n_returned"]) for c in cases)
    good = sum(int(run[c][key]["n_actionable"]) for c in cases)
    return shown, good, (good / shown if shown else 0.0)


def paired(deltas: dict[str, float]) -> str:
    vals = list(deltas.values())
    lo, hi = paired_bootstrap(vals)
    pos, neg, ties, p = sign_test(vals)
    return (
        f"{statistics.mean(vals):+.2f}/case, 95% CI [{lo:+.2f}, {hi:+.2f}], "
        f"{pos} w / {neg} l / {ties} t, sign p = {p:.4f}"
    )


def main() -> int:
    headline = load(HEADLINE_RUN)
    check_headline(headline)
    runs = {label: load(name) for label, (name, _) in DRAWS.items()}
    for label, (_, w) in DRAWS.items():
        check_draw(label, runs[label], w)
    check_same_pool(runs)

    b_net, b_failed = baseline_net(headline)
    ok_cases = sorted(b_net)
    all_cases = sorted(headline)

    print("The $0 join -- headline table at ranking.w_embedding = 1.5\n")
    print(f"  baseline    {HEADLINE_RUN}")
    print("              cached Opus 4.8 answers, cached judge verdicts, measured on pool-depth")
    print(
        f"  RepoRadar   two pre-registered draws on pool-wemb, window 15 ({len(all_cases)} cases)"
    )
    if b_failed:
        print(f"  EXCLUDED    baseline did not run on {b_failed} -- named, not scored 0")
    print()

    print("Per-arm means (RepoRadar Top Picks, net@2/case):\n")
    print(f"  {'arm':16} {'25 cases':>10} {'24 cases':>10}   (24 = where the baseline ran)")
    per_arm: dict[str, dict[str, float]] = {}
    for label in DRAWS:
        n = net(runs[label])
        per_arm[label] = n
        m25 = statistics.mean(n.values())
        m24 = statistics.mean(n[c] for c in ok_cases)
        print(f"  {label:16} {m25:>+10.2f} {m24:>+10.2f}")
    print(
        f"  {'published live':16} {HEADLINE_MEAN_25:>+10.2f} "
        f"{statistics.mean(net(headline)[c] for c in ok_cases):>+10.2f}   <- pool-depth"
    )
    print()

    # Mean-of-draws is the estimator the two-draw floor of 0.52 was derived for.
    def mean_of_draws(w: float) -> dict[str, float]:
        arms = [per_arm[label] for label, (_, ww) in DRAWS.items() if ww == w]
        return {c: statistics.mean(a[c] for a in arms) for c in all_cases}

    treat, control = mean_of_draws(1.5), mean_of_draws(0.0)

    print("Mean of two draws (the estimator the 0.52 floor was derived for):\n")
    for name, arm in (("w_embedding 0.0", control), ("w_embedding 1.5", treat)):
        print(
            f"  {name}: {statistics.mean(arm.values()):+.2f}/case over 25, "
            f"{statistics.mean(arm[c] for c in ok_cases):+.2f} over 24"
        )
    print(
        f"\n  treatment - control, paired over 25: "
        f"{paired({c: treat[c] - control[c] for c in all_cases})}"
    )
    print(f"  two-draw floor {FLOOR_TWO_DRAW} -- this is the resolved NR-38 result, unchanged.\n")

    print("Paired against the cached Opus 4.8 baseline (24 cases):\n")
    print(f"  {'arm':22} {'mean net@2':>11} {'vs Opus, paired':>18}")
    b_mean = statistics.mean(b_net.values())
    for name, arm in (("w_embedding 0.0", control), ("w_embedding 1.5", treat)):
        m = statistics.mean(arm[c] for c in ok_cases)
        print(f"  {name:22} {m:>+11.2f}   {paired({c: arm[c] - b_net[c] for c in ok_cases})}")
    print(f"  {'Opus 4.8 baseline':22} {b_mean:>+11.2f}")
    print()

    print("Volume and precision (24 baseline-ok cases, per draw -- volume does not average):\n")
    print(f"  {'arm':16} {'shown':>7} {'actionable':>11} {'precision':>10} {'net-neg repos':>14}")
    for label in DRAWS:
        shown, good, prec = volume(runs[label], ok_cases)
        negs = sum(1 for c in ok_cases if per_arm[label][c] < 0)
        print(f"  {label:16} {shown:>7} {good:>11} {prec:>10.3f} {negs:>14}")
    b_shown, b_good, b_prec = volume(headline, ok_cases, key="baseline")
    b_negs = sum(1 for c in ok_cases if b_net[c] < 0)
    print(f"  {'Opus 4.8':16} {b_shown:>7} {b_good:>11} {b_prec:>10.3f} {b_negs:>14}")
    print()

    t_mean_24 = statistics.mean(treat[c] for c in ok_cases)
    t_draws = [
        statistics.mean(per_arm[x][c] for c in ok_cases) for x in DRAWS if DRAWS[x][1] == 1.5
    ]
    print("PREDICTION for the paid arm (1.5 on pool-wemb, window 15, --baseline cli):\n")
    print("  It is a THIRD draw of the 1.5 arm, not a re-read of these two. Its own value is")
    print("  what BENCHMARK_HEADLINE will cite, wherever it lands. The NR-38 decision stays")
    print("  closed at two draws and this value is NOT averaged into it.\n")
    print(
        f"  RepoRadar net@2, 24 cases : {t_mean_24:+.2f} "
        f"(draws read {t_draws[0]:+.2f} and {t_draws[1]:+.2f})"
    )
    print(
        f"  expected range            : [{t_mean_24 - FLOOR_ONE_DRAW:+.2f}, "
        f"{t_mean_24 + FLOOR_ONE_DRAW:+.2f}]  (one-draw floor {FLOOR_ONE_DRAW})"
    )
    print(f"  paired vs Opus            : {t_mean_24 - b_mean:+.2f}/case (published at 0.0: +3.79)")
    print()
    print("  The INVARIANT under test: the paid run's baseline column must reproduce")
    print(f"  {b_mean:+.2f}/case, {b_shown} shown, {b_good} actionable EXACTLY -- same cached")
    print("  answers, same cached verdicts, different pool. Any drift there falsifies the")
    print("  join and is a finding in its own right.")
    print()
    verify(headline, b_net, b_failed)
    return 0


def verify(headline: Run, b_net: dict[str, float], b_failed: list[str]) -> None:
    """Score the prediction against the paid arm, if it has been run.

    The prediction above is worth nothing unless something checks it, and a check that
    lives in shell history is a check nobody runs twice.
    """
    path = RESULTS / PAID_ARM
    if not path.exists():
        print(f"The paid arm has not run yet ({PAID_ARM} absent).")
        return
    paid = load(PAID_ARM)
    check_draw_settings(paid)
    shared = sorted(b_net)

    print("VERIFIED against the paid arm -- " + PAID_ARM + "\n")
    p_b_net, p_b_failed = baseline_net(paid)
    pb_shown, pb_good, pb_prec = volume(paid, shared, key="baseline")
    b_shown, b_good, b_prec = volume(headline, shared, key="baseline")
    held = round(statistics.mean(p_b_net[c] for c in shared), 4) == round(
        statistics.mean(b_net.values()), 4
    ) and (pb_shown, pb_good) == (b_shown, b_good)
    print(f"  INVARIANT {'HELD' if held else '*** BROKEN ***'} on the {len(shared)} shared cases:")
    print(
        f"    published (pool-depth) : {statistics.mean(b_net.values()):+.2f}/case, "
        f"{b_shown} shown, {b_good} actionable, precision {b_prec:.3f}"
    )
    print(
        f"    paid arm  (pool-wemb)  : {statistics.mean(p_b_net[c] for c in shared):+.2f}/case, "
        f"{pb_shown} shown, {pb_good} actionable, precision {pb_prec:.3f}"
    )
    if not held:
        print("    net@2 is supposed to be pool-independent. It is not. Stop and investigate.")
    print()

    m24 = statistics.mean(net(paid)[c] for c in shared)
    predicted = statistics.mean(
        statistics.mean(per[c] for per in [net(load(n)) for n, w in DRAWS.values() if w == 1.5])
        for c in shared
    )
    inside = abs(m24 - predicted) <= FLOOR_ONE_DRAW
    print(f"  PREDICTION {'inside' if inside else 'OUTSIDE'} the one-draw floor:")
    print(
        f"    predicted {predicted:+.2f}, actual {m24:+.2f}, "
        f"miss {abs(m24 - predicted):.2f} against {FLOOR_ONE_DRAW}\n"
    )

    # The paid arm recovered `thin-lang`, so it carries a baseline on all 25. Report the
    # 24-case figure for comparability with the published +5.42 and the 25-case one as the
    # number that supersedes it -- never one silently standing in for the other [C-17].
    for label, cases in (("24 shared (comparable to +5.42)", shared), ("25 all", sorted(paid))):
        if len(cases) == 25 and p_b_failed:
            print(f"  {label}: baseline still absent on {p_b_failed}; skipped")
            continue
        rr = statistics.mean(net(paid)[c] for c in cases)
        bb = statistics.mean(net(paid, "baseline")[c] for c in cases)
        s, g, pr = volume(paid, cases)
        bs, bg, bpr = volume(paid, cases, key="baseline")
        print(f"  {label}:")
        print(
            f"    RepoRadar {rr:+.2f}  {s} shown, {g} actionable, precision {pr:.3f}, "
            f"{sum(1 for c in cases if net(paid)[c] < 0)} net-negative"
        )
        print(
            f"    Opus 4.8  {bb:+.2f}  {bs} shown, {bg} actionable, precision {bpr:.3f}, "
            f"{sum(1 for c in cases if net(paid, 'baseline')[c] < 0)} net-negative"
        )
        print(
            f"    paired    {paired({c: net(paid)[c] - net(paid, 'baseline')[c] for c in cases})}"
        )
    if b_failed and not p_b_failed:
        gained = sorted(set(b_failed) - set(p_b_failed))
        print(
            f"\n  NEW: the baseline ran on {gained} for the first time -- the published "
            f"figure excluded it."
        )


def check_draw_settings(paid: Run) -> None:
    if {r.get("w_embedding") for r in paid.values()} != {1.5}:
        raise SystemExit("the paid arm is not a w_embedding 1.5 run")
    if {r.get("digest_window") for r in paid.values()} != {15}:
        raise SystemExit("the paid arm is not at the shipped digest width")


if __name__ == "__main__":
    raise SystemExit(main())
