"""What the third judge's verdicts show beyond its registered endpoints. [NR-69]

Descriptive and post hoc, $0. Nothing here was registered in PREREG-third-judge.md, so no figure
in it confirms a hypothesis and no interval in it is a test. The registered readings are in
`evals/third_judge.json`, and nothing here changes them.

It reads two tracked artifacts:

  evals/third_judge.json       Gemini's verdicts, each with the GPT-5.5 and Sonnet labels of every
                               paper it judged
  evals/judge_dependence.json  NR-66's aug20 band rows, with each paper's admission under the
                               frozen map, and NR-66's figures for the other two judges

and writes `evals/third_judge_followups.json`. It computes three things.

1. The rescore stage's value under Gemini on the aug20 band. It uses NR-66's recipe (+1 per
   actionable and -2 per other admitted paper, set against showing none and showing all of the
   band) and NR-66's own case draws, so its interval sits beside Fig. 1's on equal terms. Before
   it computes anything for Gemini it recomputes NR-66's GPT-5.5 and Sonnet figures from the same
   rows and draws, and refuses to write unless they come back exactly.
2. Where Gemini's acceptances sit among the other two judges' on the band: a cross-tabulation and
   Cohen's kappa for each pair of judges, beside the largest kappa that pair's marginals allow.
   The GPT-5.5 and Sonnet pair must reproduce NR-68's 0.199 and 0.248.
3. The comparison under Gemini, decomposed as NR-68 decomposed it: each arm's mean net@2, the
   negative controls' share of the margin, the margin without them, and the development and later
   groups. Intervals are NR-52's own `bigram_report.paired_bootstrap`. The margin over all 37
   cases must reproduce E4 of `third_judge.json`, point and interval.

    uv run python evals/third_judge_followups.py     # offline and $0
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402
import comparison_sensitivity as cs  # noqa: E402
import finescale_model_transfer as fmt  # noqa: E402
import judge_dependence as jd  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402
from second_judge import cohens_kappa  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
THIRD = EVALS / "third_judge.json"
DEPENDENCE = EVALS / "judge_dependence.json"
OUT = EVALS / "third_judge_followups.json"

JUDGES = ("gpt", "sonnet", "gemini")
BAND = "aug20"
BAND_SIZE = 324
N_CASES = 37


def expect(ok: bool, what: str) -> None:
    """Fail loudly. `assert` would vanish under `python -O` and let a wrong artifact through."""
    if not ok:
        raise SystemExit(f"third_judge_followups: {what}")


def read(path: Path) -> Any:
    expect(path.is_file(), f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def ok(score: int) -> bool:
    return score >= tb.ACTIONABLE


# ── The band ─────────────────────────────────────────────────────────────────────────────


def band_rows(third: dict[str, Any], dependence: dict[str, Any]) -> list[dict[str, Any]]:
    """NR-66's aug20 rows, each with Gemini's score joined on (case, unversioned id).

    The GPT-5.5 and Sonnet labels that `third_judge.json` carries for each band paper must equal
    NR-66's, so the join cannot pair a verdict with the wrong paper unnoticed.
    """
    gem: dict[tuple[str, str], tuple[int, int, int]] = {}
    for r in third["rows"]:
        for m in r["members"]:
            if m["population"] == "band":
                expect(r["score"] is not None, f"band paper {m['case']}/{m['id']} is void")
                k = (m["case"], dedup_id(m["id"]))
                expect(k not in gem, f"band paper {k} appears twice")
                gem[k] = (r["score"], m["gpt"], m["sonnet"])
    rows = []
    for r in dependence["rows"]:
        if r["band"] != BAND:
            continue
        k = (r["case"], r["id"])
        expect(k in gem, f"band paper {k} has no Gemini verdict")
        score, gpt, son = gem.pop(k)
        expect((gpt, son) == (r["gpt"], r["sonnet"]), f"band paper {k} carries other labels")
        rows.append({**r, "gemini": score})
    expect(not gem, f"{len(gem)} Gemini band verdicts match no NR-66 row")
    expect(len(rows) == BAND_SIZE, f"the band has {len(rows)} rows, not {BAND_SIZE}")
    return rows


def stage(pool: list[dict[str, Any]], judge: str) -> dict[str, float]:
    """NR-66's value recipe over one sample of rows, per benchmark case."""
    adm = [r for r in pool if r["adm4o"]]
    stage_net = sum(1 if ok(r[judge]) else -2 for r in adm)
    all_net = sum(1 if ok(r[judge]) else -2 for r in pool)
    out = {
        "stage_minus_none": stage_net / N_CASES,
        "stage_minus_all": (stage_net - all_net) / N_CASES,
    }
    if adm:
        out["admitted_precision"] = sum(ok(r[judge]) for r in adm) / len(adm)
    if pool:
        out["band_actionable"] = sum(ok(r[judge]) for r in pool) / len(pool)
    return out


def stage_value(rows: list[dict[str, Any]], cases: list[str]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for r in rows:
        by_case[r["case"]].append(r)
    simple, _ = jd.draws(cases)
    out: dict[str, Any] = {}
    for judge in JUDGES:
        point = stage(rows, judge)
        boot: dict[str, list[float]] = {k: [] for k in point}
        for picked in simple:
            for k, v in stage([r for c in picked for r in by_case[c]], judge).items():
                boot[k].append(v)
        out[judge] = {
            k: {"point": point[k], "ci": list(fmt.interval(boot[k]) or ()), "draws": len(boot[k])}
            for k in point
        }
    return out


def reproduce_nr66(value: dict[str, Any], dependence: dict[str, Any]) -> None:
    """GPT-5.5 and Sonnet must come back exactly as NR-66 wrote them, point and interval."""
    nr66 = dependence["summary"]["bands"][BAND]["value"]["4o"]
    for judge in ("gpt", "sonnet"):
        for k in ("stage_minus_none", "stage_minus_all"):
            want = nr66[judge]["overall"][k]
            got = value[judge][k]
            expect(got["point"] == want["point"], f"{judge} {k} {got['point']} != NR-66")
            expect(got["ci"] == want["ci"], f"{judge} {k} interval {got['ci']} != NR-66")


def agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-tabulations and kappa with its ceiling, for each pair of judges, binary at 2."""
    out: dict[str, Any] = {}
    n = len(rows)
    for a, b in (("gpt", "sonnet"), ("gpt", "gemini"), ("sonnet", "gemini")):
        x = [1 if ok(r[a]) else 0 for r in rows]
        y = [1 if ok(r[b]) else 0 for r in rows]
        p_a, p_b = sum(x) / n, sum(y) / n
        p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
        p_o_max = min(p_a, p_b) + min(1 - p_a, 1 - p_b)
        both = sum(1 for i, j in zip(x, y, strict=True) if i and j)
        out[f"{a}_{b}"] = {
            "both": both,
            f"{a}_only": sum(x) - both,
            f"{b}_only": sum(y) - both,
            "neither": n - sum(x) - sum(y) + both,
            f"p_{a}": p_a,
            f"p_{b}": p_b,
            "kappa": cohens_kappa(x, y),
            "kappa_max": (p_o_max - p_e) / (1 - p_e),
        }
    gs = out["gpt_sonnet"]
    expect(round(gs["kappa"], 3) == cs.BAND_KAPPA, f"GPT-5.5/Sonnet kappa {gs['kappa']:.4f}")
    expect(round(gs["kappa_max"], 3) == 0.248, f"GPT-5.5/Sonnet ceiling {gs['kappa_max']:.4f}")
    gem = [r for r in rows if ok(r["gemini"])]
    out["gemini_accepts"] = {
        "n": len(gem),
        "also_gpt": sum(1 for r in gem if ok(r["gpt"])),
        "also_sonnet": sum(1 for r in gem if ok(r["sonnet"])),
        "also_both": sum(1 for r in gem if ok(r["gpt"]) and ok(r["sonnet"])),
    }
    out["gemini_scores"] = {str(s): sum(1 for r in rows if r["gemini"] == s) for s in range(4)}
    return out


# ── The comparison ───────────────────────────────────────────────────────────────────────


def margin(per_case: dict[str, dict[str, Any]], cases: list[str]) -> dict[str, Any]:
    d = [float(per_case[c]["delta"]) for c in cases]
    lo, hi = paired_bootstrap(d)
    return {
        "n": len(d),
        "margin": statistics.mean(d),
        "ci95": [lo, hi],
        "ours_mean_net": statistics.mean(per_case[c]["ours"] for c in cases),
        "baseline_mean_net": statistics.mean(per_case[c]["baseline"] for c in cases),
        "wins": sum(1 for x in d if x > 0),
        "losses": sum(1 for x in d if x < 0),
        "ties": sum(1 for x in d if x == 0),
    }


def comparison(third: dict[str, Any]) -> dict[str, Any]:
    e4 = third["summary"]["E4"]
    per_case = e4["per_case"]
    cases = sorted(per_case)
    expect(len(cases) == N_CASES, f"E4 has {len(cases)} cases")
    expect(set(cs.CONTROLS) <= set(cases), "a negative control is missing from E4")
    every = margin(per_case, cases)
    expect(every["margin"] == e4["margin"], f"E4 margin {every['margin']} != registered")
    expect(every["ci95"] == e4["ci"], f"E4 interval {every['ci95']} != registered")
    groups = cs.group_lists(cases)
    part = sum(per_case[c]["delta"] for c in cs.CONTROLS)
    return {
        "all": every,
        "controls": {
            "deltas": {c: per_case[c]["delta"] for c in cs.CONTROLS},
            "controls_sum": part,
            "total": sum(per_case[c]["delta"] for c in cases),
        },
        "without_controls": margin(per_case, [c for c in cases if c not in cs.CONTROLS]),
        "development": margin(per_case, groups["development"]),
        "later": margin(per_case, groups["later"]),
    }


# ── Writing ──────────────────────────────────────────────────────────────────────────────


def build() -> dict[str, Any]:
    third, dependence = read(THIRD), read(DEPENDENCE)
    cases = dependence["cases"][BAND]
    expect(len(cases) == N_CASES and cases == sorted(cases), "NR-66's aug20 case list")
    rows = band_rows(third, dependence)
    value = stage_value(rows, cases)
    reproduce_nr66(value, dependence)
    return {
        "_comment": (
            "Descriptive and post hoc follow-ups to NR-69, $0. Nothing here was registered. "
            "Written by evals/third_judge_followups.py and pinned by "
            "tests/test_third_judge_followups.py. Figures are stored at full precision."
        ),
        "inputs": {"third_judge": THIRD.name, "judge_dependence": DEPENDENCE.name},
        "band": BAND,
        "stage_value": {
            "recipe": "NR-66: admitted papers +1 actionable, -2 not, per benchmark case",
            "bootstrap": {"draws": jd.DRAWS, "seed": jd.SEED, "unit": "case, all 37"},
            "reproduces_nr66": True,
            "by_judge": value,
        },
        "agreement": agreement(rows),
        "comparison": {
            "bootstrap": "bigram_report.paired_bootstrap, NR-52's own",
            **comparison(third),
        },
    }


def show(art: dict[str, Any]) -> None:
    print("stage value on aug20, per repository (NR-66 draws)")
    for judge, v in art["stage_value"]["by_judge"].items():
        parts = []
        for k in ("stage_minus_none", "stage_minus_all"):
            p, (lo, hi) = v[k]["point"], v[k]["ci"]
            parts.append(f"{k} {p:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        parts.append(f"admitted precision {v['admitted_precision']['point']:.3f}")
        print(f"  {judge:<7} " + "  ".join(parts))
    a = art["agreement"]
    for pair in ("gpt_sonnet", "gpt_gemini", "sonnet_gemini"):
        print(f"  kappa {pair:<14} {a[pair]['kappa']:.3f} (max {a[pair]['kappa_max']:.3f})")
    print(f"  Gemini accepts {a['gemini_accepts']}")
    c = art["comparison"]
    for k in ("all", "without_controls", "development", "later"):
        m = c[k]
        print(
            f"  {k:<17} n={m['n']:>2} margin {m['margin']:+.2f} "
            f"[{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}]  ours {m['ours_mean_net']:+.2f} "
            f"baseline {m['baseline_mean_net']:+.2f}  {m['wins']}/{m['losses']}/{m['ties']}"
        )
    print(f"  controls {c['controls']}")


def main() -> int:
    art = build()
    OUT.write_text(json.dumps(art, indent=1) + "\n", encoding="utf-8")
    show(art)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
