"""How far does NR-52's comparison with Opus 5 depend on the penalty, the controls and the cases?

NR-52 compares the shipped arm with agentic Opus 5 on 37 cases, under three labels:

    gpt           GPT-5.5 >= 2                      margin +0.32 [-1.78, +2.51]
    consensus     GPT-5.5 >= 2 and Sonnet >= 1      margin +0.57 [-1.73, +2.92]
    sonnet_only   Sonnet >= 2                       margin -3.41 [-7.00, +0.54]

A review of that comparison found that it depends on three things the record does not state, and
that the kappa quoted beside it has a ceiling the record does not state either. This puts all four
in one tracked place with intervals, so a claim about the comparison can be checked against it and
the paper can cite it.

1. The penalty. net@lambda = a - lambda * u, where a counts shown actionable papers and u the
   rest. NR-52 stores net@2 and the shown count per case, which fix a and u exactly. The margin is
   swept over lambda in {1, 1.5, 2, 3, 4}. It is linear in lambda, so the lambda at which it
   changes sign is exact.
2. The negative controls. webdev, cli and http are repositories with no research to apply, where
   the right output is nothing. The margin is reported with and without them.
3. Development against later repositories. The 22 development cases are Testbed A's. The later 15
   are the 12 scientific cases and thin-gnn, thin-kv and thin-lang. The thin cases were added on
   2026-08-09, after Testbed A. That was before the digest width, gate depth and ranking weights
   were chosen on the 25 core cases. The 12 scientific cases came after those choices, but not
   untouched: their predictions followed a six-repository pilot, and the shipped configuration
   was confirmed with their scores visible (C-45).
4. The kappa ceiling. GPT-5.5 and Sonnet agree at kappa 0.199 on the aug20 band, binary at >= 2.
   Kappa cannot reach 1 when the two judges call different shares of the band actionable. The
   ceiling is the largest kappa those two shares allow.

It is descriptive and post hoc. The lambda grid, the controls split, the groups and the ceiling
were all chosen after NR-52's margins were known. No figure here confirms a hypothesis and no
interval here is a test. Every interval is bigram_report.paired_bootstrap on per-case deltas,
with its default draws and seed, the call NR-52's own intervals came from.

What it can settle: whether a statement about the margin keeps its sign when the penalty moves,
when the controls are removed, or when only the cases that shaped the design are left out. What it
cannot settle: which judge is right, or what penalty a user should pay for a wasted read. Nor can it
separate tuning from domain in the later group, since 12 of its 15 cases are scientific. The
crossover lambda carries no interval.

Inputs are all tracked: evals/rung1_second_judge.json, evals/judge_dependence.json and
evals/benchmark.yaml. The development case list is written out below because evals/results/ is
gitignored. When Testbed A's run file is present, the list is checked against it.

    uv run python evals/comparison_sensitivity.py     # offline and $0; writes the JSON beside it
"""

from __future__ import annotations

import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402
import bigram_report  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402
from finescale_current_gate import SCIENTIFIC  # noqa: E402
from second_judge import cohens_kappa  # noqa: E402

EVALS = Path(__file__).resolve().parent
RUNG1 = EVALS / "rung1_second_judge.json"
DEPENDENCE = EVALS / "judge_dependence.json"
BENCHMARK = EVALS / "benchmark.yaml"
OUT = EVALS / "comparison_sensitivity.json"

LABELS = ("gpt", "consensus", "sonnet_only")
LAMBDAS = (1.0, 1.5, 2.0, 3.0, 4.0)
NR52_LAMBDA = 2.0

# NR-52 as rung1_second_judge.json, its tests and RESULTS.md state it: margin and interval at
# lambda = 2. The script refuses to write if the recovered counts do not give these back.
PUBLISHED = {
    "gpt": (0.32, (-1.78, 2.51)),
    "consensus": (0.57, (-1.73, 2.92)),
    "sonnet_only": (-3.41, (-7.00, 0.54)),
}

# Read from benchmark.yaml's negative_control flag as well, and the two must agree.
CONTROLS = ("webdev", "cli", "http")

# Testbed A, band_testbeds.POOL50: the 22-case run of 2026-08-07. Written out because the run
# file lives in evals/results/, which is gitignored.
DEVELOPMENT = (
    "ann",
    "cli",
    "columnar",
    "compiler",
    "crypto",
    "cv",
    "db",
    "diffusion",
    "encryption",
    "graph",
    "http",
    "linter",
    "llminfer",
    "numerics",
    "peft",
    "rag",
    "rl",
    "speech",
    "storage",
    "systems",
    "vectordb",
    "webdev",
)
THIN = ("thin-gnn", "thin-kv", "thin-lang")
THIN_NOTE = (
    "thin-gnn, thin-kv and thin-lang were added on 2026-08-09, after the development testbed "
    "(Testbed A, 2026-08-07) but before the digest width, gate depth and ranking weights were "
    "chosen on the 25 core cases. The 12 scientific cases came after those choices but were not "
    "untouched either: their predictions followed a six-repository pilot, and the shipped "
    "configuration was confirmed with their scores visible (C-45). 'Later' is not the same "
    "as 'untouched'."
)

# The GPT-5.5 against Sonnet kappa NR-52's docstring and NR-53 quote for the aug20 band.
BAND_KAPPA = 0.199
AUG20_SIZE = 324


def expect(ok: bool, what: str) -> None:
    """Fail loudly. `assert` would vanish under `python -O` and let a wrong artifact through."""
    if not ok:
        raise SystemExit(f"comparison_sensitivity: {what}")


def read(path: Path) -> Any:
    expect(path.is_file(), f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def key(lam: float) -> str:
    return f"{lam:g}"


# ── Recovering the counts ───────────────────────────────────────────────────────────────────


def counts(net: int, n: int, where: str) -> tuple[int, int]:
    """(actionable, unactionable) from a net@2 over n shown papers.

    net = a - 2u and n = a + u, so a = (net + 2n) / 3 and u = (n - net) / 3. Both must be
    non-negative integers. If either is not, the stored pair is not a net@2 over n papers.
    """
    a3, u3 = net + 2 * n, n - net
    expect(
        a3 % 3 == 0 and u3 % 3 == 0 and a3 >= 0 and u3 >= 0,
        f"{where}: net@2 {net} over {n} papers gives no whole non-negative a and u",
    )
    return a3 // 3, u3 // 3


def recover(rung1: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    """Per label and case: each arm's actionable and unactionable counts, and the net@2 delta."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for label in LABELS:
        stored = rung1["labels"][label]["per_case"]
        rows: dict[str, dict[str, int]] = {}
        for case in sorted(stored):
            r = stored[case]
            rr_a, rr_u = counts(r["rr"], r["rr_n"], f"{label} {case} Anonymous")
            op_a, op_u = counts(r["opus5"], r["opus5_n"], f"{label} {case} Opus 5")
            delta = (rr_a - 2 * rr_u) - (op_a - 2 * op_u)
            expect(delta == r["delta"], f"{label} {case}: delta {delta} != stored {r['delta']}")
            rows[case] = {
                "rr_a": rr_a,
                "rr_u": rr_u,
                "op_a": op_a,
                "op_u": op_u,
                "delta_net2": delta,
            }
        out[label] = rows
    return out


# ── Statistics ──────────────────────────────────────────────────────────────────────────────


def net_at(a: int, u: int, lam: float) -> float:
    return a - lam * u


def deltas(rows: dict[str, dict[str, int]], cases: list[str], lam: float) -> list[float]:
    """Per-case margin at one penalty: our net@lambda minus Opus 5's."""
    return [
        float(
            net_at(rows[c]["rr_a"], rows[c]["rr_u"], lam)
            - net_at(rows[c]["op_a"], rows[c]["op_u"], lam)
        )
        for c in cases
    ]


def margin(d: list[float]) -> dict[str, Any]:
    """Mean, interval and wins, losses and ties, the way rung1_second_judge.report computes them."""
    lo, hi = paired_bootstrap(d)
    sg = tb.sign_test(d)
    return {
        "n_cases": len(d),
        "margin": statistics.mean(d),
        "ci95": [lo, hi],
        "wins": sg["pos"],
        "losses": sg["neg"],
        "ties": sg["ties"],
    }


def precision(rows: dict[str, dict[str, int]], arm: str) -> float:
    a = sum(r[f"{arm}_a"] for r in rows.values())
    return a / (a + sum(r[f"{arm}_u"] for r in rows.values()))


def penalty(rows: dict[str, dict[str, int]]) -> dict[str, Any]:
    """The margin over the lambda grid, its exact linear form, and where it changes sign.

    margin(lambda) = mean(delta a) - lambda * mean(delta u). The root is sum(delta a) /
    sum(delta u). A negative root means the margin keeps one sign for every lambda >= 0.
    """
    cases = sorted(rows)
    k = len(cases)
    sum_da = sum(r["rr_a"] - r["op_a"] for r in rows.values())
    sum_du = sum(r["rr_u"] - r["op_u"] for r in rows.values())
    root = None if sum_du == 0 else Fraction(sum_da, sum_du)
    crossover = root if root is not None and root >= 0 else None

    by_lambda: dict[str, Any] = {}
    for lam in LAMBDAS:
        d = deltas(rows, cases, lam)
        m = margin(d)
        exact = Fraction(sum_da, k) - Fraction(lam) * Fraction(sum_du, k)
        expect(abs(m["margin"] - float(exact)) < 1e-9, f"lambda {lam}: the linear form disagrees")
        rr_p, op_p = precision(rows, "rr"), precision(rows, "op")
        even = lam / (1 + lam)
        by_lambda[key(lam)] = {
            "lambda": lam,
            "rr_mean_net": statistics.mean(
                net_at(r["rr_a"], r["rr_u"], lam) for r in rows.values()
            ),
            "opus5_mean_net": statistics.mean(
                net_at(r["op_a"], r["op_u"], lam) for r in rows.values()
            ),
            **m,
            "break_even_precision": even,
            "rr_above_break_even": rr_p > even,
            "opus5_above_break_even": op_p > even,
        }

    if crossover is not None:
        above = "positive" if sum_du < 0 else "negative"
        reading = f"the margin is {above} for every lambda above {float(crossover):.4f}"
    elif sum_du == 0:
        reading = "the margin does not depend on lambda"
    else:
        sign = "positive" if sum_da > 0 else "negative"
        reading = f"the margin is {sign} for every lambda >= 0"
    return {
        "n_cases": k,
        "sum_delta_a": sum_da,
        "sum_delta_u": sum_du,
        "mean_delta_a": sum_da / k,
        "mean_delta_u": sum_du / k,
        "linear_form": (
            f"margin(lambda) = ({sum_da} {'-' if sum_du >= 0 else '+'} {abs(sum_du)} * lambda)"
            f" / {k}"
        ),
        "crossover": {
            "lambda": None if crossover is None else float(crossover),
            "exact": None if crossover is None else str(crossover),
            "root": None if root is None else float(root),
            "reading": reading,
        },
        "rr_precision": precision(rows, "rr"),
        "opus5_precision": precision(rows, "op"),
        "by_lambda": by_lambda,
    }


def controls(rows: dict[str, dict[str, int]]) -> dict[str, Any]:
    """The negative controls' part of the net@2 margin, and the margin without them."""
    cases = sorted(rows)
    kept = [c for c in cases if c not in CONTROLS]
    total = sum(rows[c]["delta_net2"] for c in cases)
    part = sum(rows[c]["delta_net2"] for c in CONTROLS)
    return {
        "deltas": {c: rows[c]["delta_net2"] for c in CONTROLS},
        "rr_shown": {c: rows[c]["rr_a"] + rows[c]["rr_u"] for c in CONTROLS},
        "opus5_shown": {c: rows[c]["op_a"] + rows[c]["op_u"] for c in CONTROLS},
        "controls_sum": part,
        "total": total,
        "share_of_total": part / total if total else None,
        "per_case_contribution": part / len(cases),
        "with": margin(deltas(rows, cases, NR52_LAMBDA)),
        "without": margin(deltas(rows, kept, NR52_LAMBDA)),
    }


def kappa_ceiling(dependence: dict[str, Any]) -> dict[str, Any]:
    """Observed Cohen's kappa on the aug20 band, binary at >= 2, and the most its marginals allow.

    kappa     = (p_o - p_e) / (1 - p_e)
    kappa_max = (p_o_max - p_e) / (1 - p_e)
    p_e       = p_gpt * p_sonnet + (1 - p_gpt) * (1 - p_sonnet)
    p_o_max   = min(p_gpt, p_sonnet) + min(1 - p_gpt, 1 - p_sonnet)

    p_gpt and p_sonnet are the shares each judge calls actionable. p_o_max is the most agreement
    two raters with those shares can reach, so kappa_max is 1 only when the shares are equal.
    """
    rows = [r for r in dependence["rows"] if r["band"] == "aug20"]
    expect(len(rows) == AUG20_SIZE, f"aug20 has {len(rows)} rows, not {AUG20_SIZE}")
    g = [1 if r["gpt"] >= tb.ACTIONABLE else 0 for r in rows]
    s = [1 if r["sonnet"] >= tb.ACTIONABLE else 0 for r in rows]
    n = len(rows)
    both = sum(1 for x, y in zip(g, s, strict=True) if x and y)
    gpt_only = sum(1 for x, y in zip(g, s, strict=True) if x and not y)
    sonnet_only = sum(1 for x, y in zip(g, s, strict=True) if y and not x)
    neither = n - both - gpt_only - sonnet_only
    p_g, p_s = sum(g) / n, sum(s) / n
    p_o = (both + neither) / n
    p_e = p_g * p_s + (1 - p_g) * (1 - p_s)
    p_o_max = min(p_g, p_s) + min(1 - p_g, 1 - p_s)
    kappa = cohens_kappa(g, s)
    expect(abs(kappa - (p_o - p_e) / (1 - p_e)) < 1e-12, "cohens_kappa disagrees with its formula")
    expect(round(kappa, 3) == BAND_KAPPA, f"aug20 kappa {kappa:.4f} is not the quoted {BAND_KAPPA}")
    ceiling = (p_o_max - p_e) / (1 - p_e)
    return {
        "band": "aug20",
        "n": n,
        "cut": f"label >= {tb.ACTIONABLE}, for both judges",
        "table": {
            "both": both,
            "gpt_only": gpt_only,
            "sonnet_only": sonnet_only,
            "neither": neither,
        },
        "p_gpt": p_g,
        "p_sonnet": p_s,
        "p_o": p_o,
        "p_e": p_e,
        "p_o_max": p_o_max,
        "kappa": kappa,
        "kappa_max": ceiling,
        "kappa_over_max": kappa / ceiling,
        "formula": (
            "kappa = (p_o - p_e) / (1 - p_e); kappa_max = (p_o_max - p_e) / (1 - p_e); "
            "p_e = p_gpt * p_sonnet + (1 - p_gpt) * (1 - p_sonnet); "
            "p_o_max = min(p_gpt, p_sonnet) + min(1 - p_gpt, 1 - p_sonnet)"
        ),
        "estimator": "second_judge.cohens_kappa, the one second_judge_band.py used",
    }


# ── Groups ──────────────────────────────────────────────────────────────────────────────────


def group_lists(cases: list[str]) -> dict[str, list[str]]:
    everything = set(cases)
    dev = set(DEVELOPMENT)
    groups = {
        "development": sorted(dev),
        "later": sorted(everything - dev),
        "scientific": sorted(SCIENTIFIC),
        "core": sorted(everything - SCIENTIFIC),
    }
    expect(len(DEVELOPMENT) == len(dev) == 22 and dev <= everything, "the development list")
    expect(set(groups["later"]) == SCIENTIFIC | set(THIN), "later is not scientific plus thin")
    expect(set(groups["core"]) == dev | set(THIN), "core is not development plus thin")
    expect(
        len(groups["scientific"]) == 12 and SCIENTIFIC.issubset(everything), "the scientific list"
    )
    return groups


def check_development_list() -> bool:
    """Against Testbed A's run file, when a checkout has it. Returns whether it was checked."""
    if not tb.POOL50.is_file():
        return False
    run = json.loads(tb.POOL50.read_text(encoding="utf-8"))
    found = {e["case"] for e in run}
    expect(found == set(DEVELOPMENT), f"Testbed A holds {sorted(found)}, not the written list")
    return True


def check_controls() -> None:
    bench = yaml.safe_load(BENCHMARK.read_text(encoding="utf-8"))
    flagged = {c["name"] for c in bench["cases"] if c.get("negative_control")}
    expect(flagged == set(CONTROLS), f"benchmark.yaml flags {sorted(flagged)} as controls")


# ── Reproduction ────────────────────────────────────────────────────────────────────────────


def reproduce(summary: dict[str, Any], rung1: dict[str, Any]) -> None:
    """Refuse to write unless lambda = 2 gives back NR-52 exactly as it was published."""
    for label, (want, (lo, hi)) in PUBLISHED.items():
        got = summary["penalty"][label]["by_lambda"][key(NR52_LAMBDA)]
        stored = rung1["labels"][label]
        expect(round(got["margin"], 2) == want, f"{label} margin {got['margin']:.4f} != {want}")
        expect(round(got["margin"], 2) == stored["margin"], f"{label} margin != stored")
        ci = [round(x, 2) for x in got["ci95"]]
        expect(ci == [lo, hi], f"{label} interval {ci} != published [{lo}, {hi}]")
        expect(ci == stored["ci95"], f"{label} interval {ci} != stored {stored['ci95']}")
        wlt = (got["wins"], got["losses"], got["ties"])
        expect(wlt == (stored["wins"], stored["losses"], stored["ties"]), f"{label} w/l/t {wlt}")
        expect(round(got["rr_mean_net"], 2) == stored["rr_mean_net2"], f"{label} Anonymous mean")
        expect(round(got["opus5_mean_net"], 2) == stored["opus5_mean_net2"], f"{label} Opus 5")
        same = summary["controls"][label]["with"]
        expect(same["margin"] == got["margin"] and same["ci95"] == got["ci95"], f"{label} with")


# ── Printing ────────────────────────────────────────────────────────────────────────────────


def ci(e: dict[str, Any]) -> str:
    return f"[{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]"


def wlt(e: dict[str, Any]) -> str:
    return f"{e['wins']}/{e['losses']}/{e['ties']}"


def show(summary: dict[str, Any]) -> None:
    print("\n1. The penalty: net@lambda = a - lambda * u, margin = ours minus Opus 5 per case")
    for label in LABELS:
        p = summary["penalty"][label]
        c = p["crossover"]
        print(
            f"\n  {label}: pooled precision Anonymous {p['rr_precision']:.3f}, "
            f"Opus 5 {p['opus5_precision']:.3f}"
        )
        print(
            f"  {'lambda':>6}{'Anonymous':>11}{'Opus 5':>9}{'margin':>9}{'CI95':>19}"
            f"{'w/l/t':>10}{'break-even':>12}{'RR above':>10}{'O5 above':>10}"
        )
        for e in p["by_lambda"].values():
            print(
                f"  {e['lambda']:>6g}{e['rr_mean_net']:>+11.2f}{e['opus5_mean_net']:>+9.2f}"
                f"{e['margin']:>+9.2f}{ci(e):>19}{wlt(e):>10}{e['break_even_precision']:>12.3f}"
                f"{e['rr_above_break_even']!s:>10}{e['opus5_above_break_even']!s:>10}"
            )
        root = "none" if c["root"] is None else f"{c['root']:+.4f}"
        cross = "none" if c["lambda"] is None else f"{c['exact']} = {c['lambda']:.4f}"
        print(
            f"  {p['linear_form']}; mean delta a {p['mean_delta_a']:+.3f}, "
            f"mean delta u {p['mean_delta_u']:+.3f}"
        )
        print(f"  crossover lambda*: {cross} (root {root}); {c['reading']}")

    print("\n2. The negative controls, net@2")
    for label in LABELS:
        c = summary["controls"][label]
        parts = ", ".join(
            f"{k} {v:+d} (shown {c['rr_shown'][k]} vs {c['opus5_shown'][k]})"
            for k, v in c["deltas"].items()
        )
        share = "n/a" if c["share_of_total"] is None else f"{c['share_of_total']:.3f}"
        print(
            f"  {label:<12} {parts}; controls {c['controls_sum']:+d} of total {c['total']:+d} "
            f"(share {share}, {c['per_case_contribution']:+.2f}/case)"
        )
        w, wo = c["with"], c["without"]
        print(
            f"  {'':<12} with {w['margin']:+.2f} {ci(w)} {wlt(w)} n={w['n_cases']}"
            f"   without {wo['margin']:+.2f} {ci(wo)} {wlt(wo)} n={wo['n_cases']}"
        )

    print("\n3. Development against later repositories, net@2")
    print(f"  {'group':<12}{'n':>4}  " + "".join(f"{label:<34}" for label in LABELS))
    for group in ("development", "later", "scientific", "core"):
        cells = []
        for label in LABELS:
            e = summary["groups"][label][group]
            cells.append(f"{e['margin']:+.2f} {ci(e)} {wlt(e)}")
        n = summary["groups"]["gpt"][group]["n_cases"]
        print(f"  {group:<12}{n:>4}  " + "".join(f"{x:<34}" for x in cells))

    k = summary["kappa"]
    t = k["table"]
    print("\n4. The kappa ceiling on the aug20 band, binary at >= 2")
    print(
        f"  n {k['n']}: both {t['both']}, gpt only {t['gpt_only']}, sonnet only "
        f"{t['sonnet_only']}, neither {t['neither']}"
    )
    print(
        f"  p_gpt {k['p_gpt']:.4f}  p_sonnet {k['p_sonnet']:.4f}  p_o {k['p_o']:.4f}  "
        f"p_e {k['p_e']:.4f}  p_o_max {k['p_o_max']:.4f}"
    )
    print(
        f"  kappa {k['kappa']:.4f}  kappa_max {k['kappa_max']:.4f}  "
        f"kappa / kappa_max {k['kappa_over_max']:.3f}"
    )
    print(f"  {k['formula']}")


def main() -> int:
    rung1 = read(RUNG1)
    dependence = read(DEPENDENCE)
    check_controls()
    checked = check_development_list()

    per_case = recover(rung1)
    cases = sorted(per_case["gpt"])
    expect(len(cases) == rung1["n_cases"] == 37, f"NR-52 holds {len(cases)} cases, not 37")
    for label in LABELS:
        expect(sorted(per_case[label]) == cases, f"{label} does not hold the same cases")
        rows = per_case[label]
        for arm, field in (("rr", "rr_papers_scored"), ("op", "opus5_papers_scored")):
            shown = sum(r[f"{arm}_a"] + r[f"{arm}_u"] for r in rows.values())
            expect(shown == rung1["labels"][label][field], f"{label} {arm} shows {shown}")
    groups = group_lists(cases)

    summary: dict[str, Any] = {
        "what": "descriptive and post hoc; the lambda grid, the controls split, the groups and "
        "the kappa ceiling were chosen after NR-52's margins were known, so no figure here "
        "confirms a hypothesis and no interval here is a test",
        "inputs": {
            "comparison": "evals/rung1_second_judge.json labels.<label>.per_case (NR-52)",
            "kappa": "evals/judge_dependence.json rows with band == 'aug20'",
            "controls": "evals/benchmark.yaml negative_control flags",
            "development": f"band_testbeds.POOL50 ({tb.POOL50.name}), written out in the script",
        },
        "development_list_checked_against_run_file": checked,
        "labels": {
            "gpt": "GPT-5.5 >= 2",
            "consensus": "GPT-5.5 >= 2 and Sonnet >= 1",
            "sonnet_only": "Sonnet >= 2",
        },
        "recovery": "net@2 = a - 2u and n = a + u, so a = (net + 2n) / 3 and u = (n - net) / 3",
        "bootstrap": {
            "function": "bigram_report.paired_bootstrap, called as rung1_second_judge.report "
            "calls it, on per-case deltas",
            "draws": bigram_report.BOOTSTRAP_N,
            "seed": bigram_report.BOOTSTRAP_SEED,
            "unit": "benchmark case, with replacement",
            "interval": "percentile, sorted means at int(0.025n) and int(0.975n)",
            "note": "every call reseeds, so two sets of deltas of the same length are "
            "resampled with the same case indices: the lambda sweep is paired across lambda",
        },
        "wlt": "band_testbeds.sign_test: wins, losses and ties of the per-case delta",
        "lambdas": list(LAMBDAS),
        "break_even": "an arm's net@lambda per shown paper is positive when its precision "
        "exceeds lambda / (1 + lambda)",
        "penalty": {label: penalty(per_case[label]) for label in LABELS},
        "controls_note": "negative controls are repositories with no research to apply, where "
        "the right output is nothing; the margin here is net@2. share_of_total is controls_sum "
        "/ total and has no share reading when the two differ in sign, as under sonnet_only; "
        "per_case_contribution is controls_sum / 37, the part of the 37-case margin they carry",
        "controls": {label: controls(per_case[label]) for label in LABELS},
        "groups_note": THIN_NOTE,
        "groups": {
            label: {
                name: margin(deltas(per_case[label], members, NR52_LAMBDA))
                for name, members in groups.items()
            }
            for label in LABELS
        },
        "kappa": kappa_ceiling(dependence),
    }
    reproduce(summary, rung1)

    OUT.write_text(
        json.dumps(
            {
                "summary": summary,
                "per_case": per_case,
                "groups": {**groups, "_note": THIN_NOTE},
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",  # LF in the working tree too, as .gitattributes asks
    )
    print(f"{len(cases)} cases; development list checked against Testbed A's run file: {checked}")
    show(summary)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
