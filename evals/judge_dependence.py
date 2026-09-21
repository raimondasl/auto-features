"""Which claims about the score-2 band depend on the judge, the scorer or the band?

Three score-2 bands are now fully dual-judged: every paper in them carries a GPT-5.5 label and a
Sonnet label. A review of what has been said about those bands found that several figures depend
on which judge reads the labels, which model scores the band, or which band is measured. The
rescore's ordering, the share of the band that is actionable, the gap between legacy and
scientific repositories, and the value of the stage were each computed once, in different
scripts, some without an interval. This puts every such figure in one tracked place with a
case-clustered interval. A claim about ordering, level, domain or stage value can then be checked
against it, and the paper can cite it.

It is descriptive and post hoc. Nothing here was pre-registered, so no figure in it confirms a
hypothesis and no interval in it is a test. The one borrowed recipe is E4's comparison from
PREREG-finescale-model-transfer.md: band net at +1 per actionable and -2 per other admitted paper,
set against show-all and show-none, under both judges. It changes one thing. A per-repository
figure divides by every benchmark case in the cohort (37, 25 or 12), and a case with no band paper
is resampled as a zero. NR-65 divided by the cases that had band papers, so its per-case figures
are larger in size than these by 37/34 on band H and 37/35 on band L. The band totals are the
same, and this script checks that they are.

What it can settle: whether a figure keeps its sign and rough size when the judge, the scorer or
the band changes, on these three bands. What it cannot settle: which judge is right, since no
human labels exist for these papers. Nor can it say how the stage behaves under another gate or
pool, or anything about papers the gate scored 3 or below 2.

The bands:
  aug20  the 324 band papers of the 2026-08-20 cohort-3 session (second_judge_band.py). Haiku
         gated. The runs predate the gate-provider option, so they record no gate.
  H      the 328 band papers of the 2026-09-08 sweep under the shipped Haiku gate (NR-64).
  L      the 315 band papers of the same sweep under the gpt-5.6-luna gate.
The scorers are gpt-4o-mini's exp09 on all three bands, and gpt-4.1-mini's on Azure for H and L
(NR-65, the product-parser reading). A paper is admitted when the shipped frozen map puts it at
2/3 or above.

    uv run python evals/judge_dependence.py     # offline and $0; writes judge_dependence.json
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402
import finescale_model_transfer as fmt  # noqa: E402
from finescale_current_gate import SCIENTIFIC  # noqa: E402
from finescale_domains import DEFAULT_LEGACY, DEFAULT_SCI  # noqa: E402
from rung1_second_judge import cached_sonnet, son_of  # noqa: E402
from second_judge import DEFAULT_MODEL as SONNET  # noqa: E402

from reporadar import finescale  # noqa: E402
from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
OUT = EVALS / "judge_dependence.json"
AUG20 = EVALS / ".work" / "second_judge_band.json"
TRANSFER = EVALS / "finescale_model_transfer.json"

# NR-65's seed and draw count. The draws themselves differ: NR-65 resamples only the cases that
# have band papers (34 on H, 35 on L), and this resamples all 37, empty cases counting as zero.
SEED = 20260921
DRAWS = 4000

BANDS = ("aug20", "H", "L")
# Row keys for each scorer's exp09 and its admission under the frozen map.
SCORERS = {"4o": ("s4o", "adm4o"), "41": ("s41", "adm41")}
SCORER_NAMES = {
    "4o": "gpt-4o-mini exp09",
    "41": "gpt-4.1-mini exp09 on Azure, product-parser reading (NR-65)",
}
JUDGES = ("gpt", "sonnet")
COHORTS = ("overall", "legacy", "scientific")

# Figures other artifacts already state. The script refuses to write if it cannot reproduce
# them: a mismatch means the rows here are not the rows those figures describe.
AUG20_AUC = {"gpt": 0.7287, "sonnet": 0.7017}  # second_judge_band.json, group ALL-37
AUG20_STAGE_MINUS_ALL = {  # second_judge_band.json stage_value, per case of the cohort
    ("scientific", "gpt"): -1.25,
    ("scientific", "sonnet"): 3.75,
    ("legacy", "gpt"): -0.08,
    ("legacy", "sonnet"): 2.08,
}
H_4O_GPT_AUC = 0.7257  # finescale_current_gate.json, NR-64


def expect(ok: bool, what: str) -> None:
    """Fail loudly. `assert` would vanish under `python -O` and let a wrong artifact through."""
    if not ok:
        raise SystemExit(f"judge_dependence: {what}")


def read(path: Path) -> Any:
    expect(path.is_file(), f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def cohort_of(case: str) -> str:
    return "scientific" if case in SCIENTIFIC else "legacy"


def admitted(exp09: float) -> bool:
    return fmt.admitted(exp09)


def gate_of(run: list[dict[str, Any]], label: str) -> tuple[str, str]:
    """The one gate every entry of a run resolves to, read the way NR-65 reads it (C-37)."""
    gates = {fmt.resolved_gate(e) for e in run}
    expect(len(gates) == 1, f"{label} resolves to more than one gate: {sorted(gates)}")
    return gates.pop()


def case_list(run: list[dict[str, Any]], label: str) -> list[str]:
    cases = [e["case"] for e in run]
    expect(len(cases) == len(set(cases)) == 37, f"{label} does not hold 37 distinct cases")
    return sorted(cases)


def row(
    band: str,
    case: str,
    pid: str,
    gpt: int,
    sonnet: int,
    s4o: float,
    s41: float | None,
    adm4o: bool,
    adm41: bool | None,
) -> dict[str, Any]:
    return {
        "band": band,
        "case": case,
        "id": pid,
        "cohort": cohort_of(case),
        "gpt": int(gpt),
        "sonnet": int(sonnet),
        "s4o": s4o,
        "s41": s41,
        "adm4o": adm4o,
        "adm41": adm41,
    }


# ── Loading ──────────────────────────────────────────────────────────────────────────────


def load_aug20(sonnet: dict[tuple[str, str], int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 2026-08-20 band as second_judge_band.py left it, with its stored labels and flags.

    The stored `shown` flag is the admission, because it is what that run's analysis used. It is
    recomputed from `finescale` anyway, and so is each Sonnet label from the verdict cache, so
    that a disagreement is counted rather than inherited.
    """
    data = read(AUG20)
    rows: list[dict[str, Any]] = []
    shown_disagree = cache_disagree = 0
    for r in data["rows"]:
        case = r["case"]
        expect(
            (r["population"] == "sci") == (case in SCIENTIFIC),
            f"aug20 {case} is filed as {r['population']}, against the cohort list",
        )
        shown_disagree += admitted(r["finescale"]) != bool(r["shown"])
        cache_disagree += son_of(sonnet, case, r["arxiv_id"]) != r["sonnet_score"]
        rows.append(
            row(
                "aug20",
                case,
                dedup_id(r["arxiv_id"]),
                r["gpt_score"],
                r["sonnet_score"],
                r["finescale"],
                None,
                bool(r["shown"]),
                None,
            )
        )
    # A changed verdict would mean the labels here are not the ones that run reported on.
    expect(cache_disagree == 0, f"{cache_disagree} aug20 Sonnet labels differ from the cache")
    return rows, {
        "admission_disagreements": shown_disagree,
        "sonnet_cache_disagreements": cache_disagree,
        "groups": {g["label"]: g for g in data["groups"]},
        "stage_value": data["stage_value"],
    }


def load_run_band(name: str, sonnet: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    """Band H or L: NR-64's rows, with NR-65's gpt-4.1-mini score and the cached Sonnet verdict.

    Sonnet verdicts are filed under the run's versioned id, and the rows carry the dedup id, so
    the run file supplies the map back. A band paper without a verdict is a void row, and the
    script stops rather than drop it.
    """
    band = fmt.BANDS[name]
    run = read(RESULTS / band.run)
    versioned: dict[tuple[str, str], str] = {}
    for entry in run:
        for paper in (entry.get("returned") or {}).get("reporadar_top10") or []:
            vid = str(paper.get("arxiv_id") or "")
            key = (entry["case"], dedup_id(vid))
            expect(versioned.get(key, vid) == vid, f"band {name} {key} has two versioned ids")
            versioned[key] = vid
    transfer = read(TRANSFER)["rows"]["first"]
    rows: list[dict[str, Any]] = []
    for c in read(EVALS / band.control)["rows"]:
        key = (c["case"], c["id"])
        expect(key in versioned, f"band {name} {key} is not in its run file")
        son = son_of(sonnet, c["case"], versioned[key])
        expect(son is not None, f"band {name} {key} has no Sonnet verdict: a void row")
        t = transfer.get(f"{c['case']}/{c['id']}") or {}
        s41 = t.get("product_exp")
        # Every paper got a product-parser score, so NR-65's enough_scored fallback admitted no
        # case whole. That is checked against its summary in reproduce().
        expect(s41 is not None, f"band {name} {key} has no gpt-4.1-mini product-parser score")
        expect(c["exp09"] is not None, f"band {name} {key} has no gpt-4o-mini score")
        rows.append(
            row(
                name,
                c["case"],
                c["id"],
                c["judge"],
                son,
                c["exp09"],
                s41,
                admitted(c["exp09"]),
                admitted(s41),
            )
        )
    expect(len(rows) == band.size, f"band {name} has {len(rows)} rows, registered {band.size}")
    return rows


# ── Statistics ───────────────────────────────────────────────────────────────────────────


def actionable(r: dict[str, Any], judge: str) -> bool:
    return bool(r[judge] >= tb.ACTIONABLE)


def auc_of(pool: list[dict[str, Any]], scorer: str, judge: str) -> float | None:
    labels = [actionable(r, judge) for r in pool]
    if not 0 < sum(labels) < len(labels):
        return None
    return tb.auc([r[SCORERS[scorer][0]] for r in pool], labels)


def base_of(pool: list[dict[str, Any]], judge: str) -> float | None:
    return sum(actionable(r, judge) for r in pool) / len(pool) if pool else None


def net(pool: list[dict[str, Any]], judge: str, adm_key: str | None) -> int:
    """Band net over the admitted papers, or over every paper when *adm_key* is None."""
    return sum(1 if actionable(r, judge) else -2 for r in pool if adm_key is None or r[adm_key])


def figures(
    pools: dict[str, list[dict[str, Any]]], divisors: dict[str, int], scorers: list[str]
) -> dict[tuple[str, ...], float]:
    """Every figure for one sample of cases.

    The point estimates and every bootstrap draw go through this one function, so an interval
    cannot describe a different quantity from the point it is printed beside.
    """
    out: dict[tuple[str, ...], float] = {}
    for cohort, pool in pools.items():
        for judge in JUDGES:
            b = base_of(pool, judge)
            if b is not None:
                out[("base", judge, cohort)] = b
        if ("base", "gpt", cohort) in out and ("base", "sonnet", cohort) in out:
            out[("level", cohort)] = out[("base", "gpt", cohort)] - out[("base", "sonnet", cohort)]
        for scorer in scorers:
            adm = SCORERS[scorer][1]
            for judge in JUDGES:
                a = auc_of(pool, scorer, judge)
                if a is not None:
                    out[("auc", scorer, judge, cohort)] = a
                stage, every = net(pool, judge, adm), net(pool, judge, None)
                out[("minus_none", scorer, judge, cohort)] = stage / divisors[cohort]
                out[("minus_all", scorer, judge, cohort)] = (stage - every) / divisors[cohort]
            g = out.get(("auc", scorer, "gpt", cohort))
            s = out.get(("auc", scorer, "sonnet", cohort))
            if g is not None and s is not None:
                out[("ordering", scorer, cohort)] = g - s
    if "legacy" in pools and "scientific" in pools:
        for scorer in scorers:
            for judge in JUDGES:
                leg = out.get(("auc", scorer, judge, "legacy"))
                sci = out.get(("auc", scorer, judge, "scientific"))
                if leg is not None and sci is not None:
                    out[("cohort_gap", scorer, judge)] = leg - sci
        # Does the gap depend on the judge? A cell whose interval spans zero cannot say a gap is
        # absent, only that it is not detected; the paired contrast on the same draw can.
        for scorer in scorers:
            g = out.get(("cohort_gap", scorer, "gpt"))
            s = out.get(("cohort_gap", scorer, "sonnet"))
            if g is not None and s is not None:
                out[("gap_judge_contrast", scorer)] = g - s
        if len(scorers) == 2:
            for judge in JUDGES:
                a = out.get(("cohort_gap", "4o", judge))
                b = out.get(("cohort_gap", "41", judge))
                if a is not None and b is not None:
                    out[("did", judge)] = a - b
    return out


def score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where each scorer puts one band's papers on the 0-9 scale, relative to the frozen cut.

    NR-65 described gpt-4.1-mini as "slightly lower and wider" and its extra admissions as coming
    from spread. The shape is the other way round, and the admission count depends on it, so the
    figures that say so are kept here, with the cut they are read against.
    """
    t = finescale.SHOW_THRESHOLD
    cut = (math.log(t / (1 - t)) - finescale.INTERCEPT) / finescale.SLOPE
    out: dict[str, Any] = {"cut": cut, "n": len(rows)}
    for key in ("4o", "41"):
        v = sorted(r[SCORERS[key][0]] for r in rows)
        q1, med, q3 = statistics.quantiles(v, n=4, method="inclusive")
        out[key] = {
            "mean": statistics.fmean(v),
            "sd": statistics.stdev(v),
            "q1": q1,
            "median": med,
            "q3": q3,
            "iqr": q3 - q1,
            "max": v[-1],
            "below_6": sum(1 for x in v if x < 6.0),
            "from_6_to_cut": sum(1 for x in v if 6.0 <= x < cut),
            "from_cut_to_7_5": sum(1 for x in v if cut <= x < 7.5),
            "from_7_5_to_8": sum(1 for x in v if 7.5 <= x < 8.0),
            "at_or_above_8": sum(1 for x in v if x >= 8.0),
        }
    out["between_scorer_spearman"] = fmt.spearman(
        [r[SCORERS["4o"][0]] for r in rows], [r[SCORERS["41"][0]] for r in rows]
    )
    return out


def draws(cases: list[str]) -> tuple[list[list[str]], list[tuple[list[str], list[str]]]]:
    """The overall draws, and the stratified ones for anything read within a cohort.

    An overall draw picks 37 cases from the 37. NR-65 drew only the cases with band papers, so
    only the seed and draw count are shared with it. A cohort figure resamples only its
    own cohort's cases, so a per-repository figure keeps its 25 or 12 cases in every draw, and a
    legacy-minus-scientific gap never loses a cohort to chance. Each set gets its own
    random.Random(SEED), so adding a statistic cannot move the draws of another.
    """
    rng = random.Random(SEED)
    simple = [[rng.choice(cases) for _ in cases] for _ in range(DRAWS)]
    legacy = [c for c in cases if c not in SCIENTIFIC]
    sci = [c for c in cases if c in SCIENTIFIC]
    rng = random.Random(SEED)
    stratified = [
        ([rng.choice(legacy) for _ in legacy], [rng.choice(sci) for _ in sci]) for _ in range(DRAWS)
    ]
    return simple, stratified


def analyse(
    rows: list[dict[str, Any]], cases: list[str], scorers: list[str]
) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], list[float]]]:
    by_case: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for r in rows:
        expect(r["case"] in by_case, f"band case {r['case']} is not a benchmark case")
        by_case[r["case"]].append(r)
    n_leg = sum(1 for c in cases if c not in SCIENTIFIC)
    divisors = {"overall": len(cases), "legacy": n_leg, "scientific": len(cases) - n_leg}

    def pooled(picked: list[str]) -> list[dict[str, Any]]:
        return [r for c in picked for r in by_case[c]]

    point = figures({"overall": rows}, divisors, scorers)
    point.update(
        figures(
            {
                "legacy": [r for r in rows if r["cohort"] == "legacy"],
                "scientific": [r for r in rows if r["cohort"] == "scientific"],
            },
            divisors,
            scorers,
        )
    )
    boot: dict[tuple[str, ...], list[float]] = {k: [] for k in point}
    simple, stratified = draws(cases)
    for picked in simple:
        for k, v in figures({"overall": pooled(picked)}, divisors, scorers).items():
            boot.setdefault(k, []).append(v)
    for leg, sci in stratified:
        drawn = {"legacy": pooled(leg), "scientific": pooled(sci)}
        for k, v in figures(drawn, divisors, scorers).items():
            boot.setdefault(k, []).append(v)
    return point, boot


# ── The artifact ─────────────────────────────────────────────────────────────────────────


def r4(x: float | None) -> float | None:
    """Stored at full precision, deliberately.

    The first version rounded to 4 dp here and printed 3 dp from that, so any figure whose 4-dp
    form ended in 5 was rounded twice: 283/324 = 0.87346 stored as 0.8735 printed as 0.874. Five
    figures in the first draft of NR-66 carried that error. Rounding belongs to whoever prints.
    """
    return x


def est(point: dict, boot: dict, key: tuple[str, ...]) -> dict[str, Any]:
    ci = fmt.interval(boot.get(key, []))
    return {
        "point": r4(point.get(key)),
        "ci": None if ci is None else [r4(ci[0]), r4(ci[1])],
        "skipped_draws": DRAWS - len(boot.get(key, [])),
    }


def band_summary(
    rows: list[dict[str, Any]], point: dict, boot: dict, cases: list[str], scorers: list[str]
) -> dict[str, Any]:
    pools = {
        "overall": rows,
        "legacy": [r for r in rows if r["cohort"] == "legacy"],
        "scientific": [r for r in rows if r["cohort"] == "scientific"],
    }
    n_leg = sum(1 for c in cases if c not in SCIENTIFIC)
    divisors = {"overall": len(cases), "legacy": n_leg, "scientific": len(cases) - n_leg}
    out: dict[str, Any] = {"auc": {}, "cohort_gap": {}, "judge_gap": {}, "value": {}}
    for scorer in scorers:
        adm = SCORERS[scorer][1]
        out["auc"][scorer] = {
            judge: {
                cohort: {
                    **est(point, boot, ("auc", scorer, judge, cohort)),
                    "n": len(pools[cohort]),
                    "base_rate": r4(point.get(("base", judge, cohort))),
                }
                for cohort in COHORTS
            }
            for judge in JUDGES
        }
        out["cohort_gap"][scorer] = {
            judge: est(point, boot, ("cohort_gap", scorer, judge)) for judge in JUDGES
        }
        out["cohort_gap"][scorer]["gpt_minus_sonnet"] = est(
            point, boot, ("gap_judge_contrast", scorer)
        )
        out["judge_gap"][scorer] = {
            cohort: {
                "ordering": est(point, boot, ("ordering", scorer, cohort)),
                "level": est(point, boot, ("level", cohort)),
            }
            for cohort in COHORTS
        }
        out["value"][scorer] = {}
        for judge in JUDGES:
            out["value"][scorer][judge] = {}
            for cohort in COHORTS:
                pool, n = pools[cohort], divisors[cohort]
                shown = [r for r in pool if r[adm]]
                stage, every = net(pool, judge, adm), net(pool, judge, None)
                out["value"][scorer][judge][cohort] = {
                    "n_cases": n,
                    "admitted": len(shown),
                    "precision": r4(
                        sum(actionable(r, judge) for r in shown) / len(shown) if shown else None
                    ),
                    "totals": {"stage": stage, "show_all": every, "show_none": 0},
                    "per_repository": {
                        "stage": r4(stage / n),
                        "show_all": r4(every / n),
                        "show_none": 0.0,
                    },
                    "stage_minus_none": est(point, boot, ("minus_none", scorer, judge, cohort)),
                    "stage_minus_all": est(point, boot, ("minus_all", scorer, judge, cohort)),
                }
    if len(scorers) == 2:
        out["did"] = {judge: est(point, boot, ("did", judge)) for judge in JUDGES}
    return out


def reproduce(summary: dict[str, Any], aug20: dict[str, Any]) -> None:
    """Refuse to write unless the figures other artifacts state come back out of these rows."""
    a = summary["bands"]["aug20"]
    for judge, want in AUG20_AUC.items():
        got = a["auc"]["4o"][judge]["overall"]["point"]
        stored = aug20["groups"]["ALL-37"][f"auc_{judge}"]
        expect(abs(got - want) <= 1e-4, f"aug20 {judge} AUC {got} is not {want}")
        expect(abs(got - stored) <= 1e-4, f"aug20 {judge} AUC {got} is not the stored {stored}")
    names = {"gpt": "GPT-5.5", "sonnet": "Sonnet"}
    pops = {"scientific": "sci", "legacy": "legacy"}
    for (cohort, judge), want in AUG20_STAGE_MINUS_ALL.items():
        got = a["value"]["4o"][judge][cohort]["stage_minus_all"]["point"]
        stored = aug20["stage_value"][pops[cohort]][names[judge]]["per_case"]
        expect(abs(got - want) <= 1e-4, f"aug20 {cohort} {judge} stage-minus-all {got} != {want}")
        expect(abs(got - stored) <= 1e-4, f"aug20 {cohort} {judge} {got} != stored {stored}")

    # NR-64's own summaries, for both run bands.
    for name in ("H", "L"):
        control = read(EVALS / fmt.BANDS[name].control)["summary"]
        for cohort in COHORTS:
            got = summary["bands"][name]["auc"]["4o"]["gpt"][cohort]["point"]
            want = control[cohort]["auc"]
            expect(abs(got - want) <= 1e-4, f"band {name} {cohort} 4o AUC {got} != NR-64 {want}")
    got = summary["bands"]["H"]["auc"]["4o"]["gpt"]["overall"]["point"]
    expect(abs(got - H_4O_GPT_AUC) <= 1e-4, f"band H 4o GPT AUC {got} is not {H_4O_GPT_AUC}")

    # NR-65's product-parser reading: AUCs, the fallback, and E4's band totals.
    nr65 = read(TRANSFER)["summary"]["bands"]
    for name in ("H", "L"):
        reading = nr65[name]["readings"]["product_parser"]
        expect(reading["fallback_cases"] == [], f"NR-65 band {name} has fallback cases")
        b = summary["bands"][name]
        for key, scorer, judge in (
            ("auc_c_gpt", "4o", "gpt"),
            ("auc_t_gpt", "41", "gpt"),
            ("auc_c_son", "4o", "sonnet"),
            ("auc_t_son", "41", "sonnet"),
        ):
            got, want = b["auc"][scorer][judge]["overall"]["point"], reading["point"][key]
            expect(abs(got - want) <= 1e-4, f"band {name} {key} {got} != NR-65 {want}")
        for judge, short in (("gpt", "gpt"), ("sonnet", "son")):
            totals = reading["band_net_totals"][short]
            for key, scorer, part in (
                ("treatment", "41", "stage"),
                ("control", "4o", "stage"),
                ("show_all", "4o", "show_all"),
            ):
                got = b["value"][scorer][judge]["overall"]["totals"][part]
                expect(got == totals[key], f"band {name} {judge} {key} {got} != NR-65 {totals}")


# ── Printing ─────────────────────────────────────────────────────────────────────────────


def cell(e: dict[str, Any], digits: int = 3) -> str:
    if e["point"] is None:
        return "n/a"
    ci = e["ci"]
    span = "[n/a]" if ci is None else f"[{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}]"
    skipped = f" skip {e['skipped_draws']}" if e["skipped_draws"] else ""
    return f"{e['point']:+.{digits}f} {span}{skipped}"


def show(summary: dict[str, Any]) -> None:
    bands = summary["bands"]
    print("\nAUC of exp09 against each judge's actionable label (case bootstrap, 95%)")
    print(f"  {'band':5} {'scorer':6} {'judge':6}  " + "  ".join(f"{c:<31}" for c in COHORTS))
    for name in BANDS:
        for scorer, by_judge in bands[name]["auc"].items():
            for judge in JUDGES:
                cells = [f"{cell(by_judge[judge][c])} n={by_judge[judge][c]['n']}" for c in COHORTS]
                print(f"  {name:5} {scorer:6} {judge:6}  " + "  ".join(f"{x:<31}" for x in cells))
    print("\nBase rate (share actionable)")
    for name in BANDS:
        by_judge = bands[name]["auc"]["4o"]
        parts = [
            f"{judge} " + " / ".join(f"{by_judge[judge][c]['base_rate']:.3f}" for c in COHORTS)
            for judge in JUDGES
        ]
        print(f"  {name:5} overall/legacy/scientific   " + "   ".join(parts))

    print("\nCohort gap: AUC(legacy) - AUC(scientific), stratified bootstrap")
    for name in BANDS:
        for scorer, by_judge in bands[name]["cohort_gap"].items():
            print(
                f"  {name:5} {scorer:3} "
                + "   ".join(f"{judge} {cell(by_judge[judge])}" for judge in JUDGES)
                + f"   gpt-sonnet {cell(by_judge['gpt_minus_sonnet'])}"
            )
        if "did" in bands[name]:
            did = bands[name]["did"]
            print(
                f"  {name:5} did gap(4o) - gap(41)   "
                + "   ".join(f"{judge} {cell(did[judge])}" for judge in JUDGES)
            )

    print("\nJudge gap: ordering = AUC(gpt) - AUC(sonnet); level = base(gpt) - base(sonnet)")
    for name in BANDS:
        for scorer, by_cohort in bands[name]["judge_gap"].items():
            for what in ("ordering", "level"):
                if what == "level" and scorer != "4o":
                    continue  # the level does not depend on the scorer
                print(
                    f"  {name:5} {scorer if what == 'ordering' else '-':3} {what:8} "
                    + "   ".join(f"{c} {cell(by_cohort[c][what])}" for c in COHORTS)
                )

    print("\nStage value per repository: stage - show_none and stage - show_all (net@2, +1/-2)")
    for name in BANDS:
        for scorer, by_judge in bands[name]["value"].items():
            for judge in JUDGES:
                for c in COHORTS:
                    v = by_judge[judge][c]
                    prec = "n/a" if v["precision"] is None else f"{v['precision']:.3f}"
                    print(
                        f"  {name:5} {scorer:3} {judge:6} {c:10} /{v['n_cases']:<2}"
                        f" -none {cell(v['stage_minus_none'], 2):<28}"
                        f" -all {cell(v['stage_minus_all'], 2):<28}"
                        f" admitted {v['admitted']:3d} precision {prec}"
                    )


def main() -> int:
    sonnet = cached_sonnet(SONNET)
    aug20_rows, aug20 = load_aug20(sonnet)
    band_rows = {
        "aug20": aug20_rows,
        "H": load_run_band("H", sonnet),
        "L": load_run_band("L", sonnet),
    }

    runs = {name: read(RESULTS / fmt.BANDS[name].run) for name in ("H", "L")}
    cases = {name: case_list(runs[name], f"run {name}") for name in ("H", "L")}
    # The 2026-08-20 session ran its cohorts as two files. Their union must be the same 37 cases,
    # or dividing aug20 by band H's case list would be dividing by the wrong benchmark.
    aug20_runs = [read(RESULTS / f) for f in (DEFAULT_SCI, DEFAULT_LEGACY)]
    expect(
        case_list(aug20_runs[0] + aug20_runs[1], "the aug20 runs") == cases["H"],
        "the aug20 runs do not hold band H's 37 cases",
    )
    cases["aug20"] = cases["H"]

    recorded = gate_of(aug20_runs[0] + aug20_runs[1], "the aug20 runs")
    expect(recorded == ("", ""), f"the aug20 runs now record a gate: {recorded}")
    gates: dict[str, dict[str, str]] = {
        "aug20": {
            "provider": "claude",
            "model": "claude-haiku-4-5",
            "read_from": (
                "not recorded: the 2026-08-20 runs have no ranking_config or pool_config, "
                "because they predate the gate-provider option (be986c7). The gate was the "
                "default Claude triage model, claude-haiku-4-5."
            ),
        }
    }
    for name in ("H", "L"):
        provider, model = gate_of(runs[name], f"run {name}")
        expect((provider, model) == fmt.BANDS[name].gate, f"run {name} gate {provider}/{model}")
        # Say which field named the model. H leaves rr_gate_model empty, so its model is the
        # rr_triage_model fallback, and a reader checking the run file should know to look there.
        explicit = bool((runs[name][0].get("ranking_config") or {}).get("rr_gate_model"))
        gates[name] = {
            "provider": provider,
            "model": model,
            "read_from": "ranking_config.rr_gate_provider and rr_gate_model"
            if explicit
            else "ranking_config.rr_gate_provider; rr_gate_model is empty there, so the model "
            "is the pool_config.rr_triage_model fallback, as resolved_gate reads it",
        }

    summary: dict[str, Any] = {
        "what": "descriptive and post hoc; nothing here was pre-registered except that the E4 "
        "comparison recipe mirrors PREREG-finescale-model-transfer.md",
        "inputs": {
            "aug20": {
                "band": "evals/.work/second_judge_band.json",
                "runs": [DEFAULT_SCI, DEFAULT_LEGACY],
                "case_list": fmt.BANDS["H"].run,
            },
            **{
                name: {
                    "run": fmt.BANDS[name].run,
                    "gpt_4o_mini_and_gpt_labels": f"evals/{fmt.BANDS[name].control}",
                    "gpt_4_1_mini": "evals/finescale_model_transfer.json",
                    "case_list": fmt.BANDS[name].run,
                }
                for name in ("H", "L")
            },
            "sonnet": f"evals/.work/second_judge/{SONNET} via rung1_second_judge.cached_sonnet",
        },
        "gpt_4_1_mini_key": (
            'finescale_model_transfer.json rows.first["<case>/<id>"].product_exp, the '
            "product-parser reading an Azure user runs; its AUCs are "
            "summary.bands.<B>.readings.product_parser.point.auc_t_gpt and auc_t_son"
        ),
        "scorers": SCORER_NAMES,
        "judges": {"gpt": "GPT-5.5 (the benchmark judge)", "sonnet": SONNET},
        "gate": gates,
        "map": {
            "slope": finescale.SLOPE,
            "intercept": finescale.INTERCEPT,
            "threshold": finescale.SHOW_THRESHOLD,
            "rule": "admitted when reporadar.finescale.probability(exp09) >= threshold",
        },
        "actionable": f"label >= {tb.ACTIONABLE}, for both judges",
        "estimator": "band_testbeds.auc",
        "bootstrap": {
            "seed": SEED,
            "draws": DRAWS,
            "unit": "benchmark case, with replacement; cases with no band paper are drawn too",
            "overall": "37 cases from the 37; NR-65 shares the seed and draw count but drew only "
            "the cases with band papers",
            "cohort": "stratified: 25 legacy from the 25 and 12 scientific from the 12, "
            "for every cohort figure, cohort gap and difference in differences",
            "interval": "percentile, sorted values at int(0.025k) and int(0.975k) - 1",
            "skipped": "a draw is skipped for a statistic only when a label set is degenerate",
        },
        "value_recipe": "stage: admitted papers +1 actionable, -2 not; show_all: every band "
        "paper; show_none: 0; per repository: total / benchmark cases in the cohort "
        "(37 overall, 25 legacy, 12 scientific)",
        "aug20_admission": {
            "used": "the stored shown flag",
            "disagreements_with_recomputed": aug20["admission_disagreements"],
        },
        "aug20_sonnet_cache_disagreements": aug20["sonnet_cache_disagreements"],
        "counts": {},
        "bands": {},
    }
    for name in BANDS:
        rows = band_rows[name]
        scorers = ["4o"] if name == "aug20" else ["4o", "41"]
        point, boot = analyse(rows, cases[name], scorers)
        summary["counts"][name] = {
            "papers": len(rows),
            "legacy": sum(1 for r in rows if r["cohort"] == "legacy"),
            "scientific": sum(1 for r in rows if r["cohort"] == "scientific"),
            "cases_with_band_papers": len({r["case"] for r in rows}),
        }
        summary["bands"][name] = band_summary(rows, point, boot, cases[name], scorers)
        if name != "aug20":
            summary["bands"][name]["score_distribution"] = score_distribution(rows)
        print(f"band {name}: {len(rows)} papers, bootstrap done", flush=True)

    reproduce(summary, aug20)
    everything = [r for name in BANDS for r in band_rows[name]]
    OUT.write_text(
        json.dumps(
            {"summary": summary, "rows": everything, "cases": {n: cases[n] for n in BANDS}},
            indent=1,
        ),
        encoding="utf-8",
    )
    show(summary)
    print(
        f"\naug20 admission: stored flag used; {aug20['admission_disagreements']} disagree with "
        f"the recomputed map. Map slope {finescale.SLOPE}, intercept {finescale.INTERCEPT}, "
        f"threshold {finescale.SHOW_THRESHOLD:.6f}."
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
