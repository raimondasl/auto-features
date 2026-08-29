"""NR-42: would a filter on non-arXiv material pay, and does any instrument support one?

A $0 probe over artifacts already on disk -- the three same-day source arms and the frozen
pools they ranked. No LLM calls, no judge calls, nothing re-run.

The item under test is the "relevance condition on non-arXiv results" that P24 retired, C-33
reopened in a narrower form, and C-34 then aimed at the wrong term. This probe asks four
questions and answers all four against it:

1. **What would it filter?** The source's own papers -- and C-34 established that term is
   already POSITIVE on both sources (+0.73 Europe PMC, +0.46 OpenAlex). A filter can only
   shrink a positive quantity unless it discriminates almost perfectly.
2. **Can anything we compute discriminate?** The gate's score and the fine-scale rescore are
   the only two pre-judge signals in the system, and they are the two stages that solved this
   exact problem for arXiv papers. On non-arXiv papers the gate-3 rate is **0.588 among
   actionable and 0.588 among non-actionable**, and the rescore's mean P is 0.842 against
   0.850 -- the wrong way round.
3. **Is the cost even in that term?** No. 64% of OpenAlex's -1.22 displacement is arXiv
   papers losing their place, and not in the digest -- only 3-5 of 37 cases reach the
   15-paper window. They lose it in the gate's input, `gate_depth: 50`, shared across sources.
4. **What IS wrong, then?** A quarter of OpenAlex candidates arrive with **no abstract**,
   where Europe PMC's are 100% complete. The gate and the rescore both read
   `paper["abstract"]` with no guard, so those papers are scored on their titles. Among shown
   papers, 4 of 17 non-actionable have no abstract against 1 of 51 actionable.

That last one is a real defect, and it is not a relevance defect. A paper with no abstract is
not an irrelevant paper; it is an **unmeasured** one, and scoring it anyway is void read as
signal -- this project's most repeated failure (C-4, C-30, the 21% that measured nothing, the
pool scanner that read 1,250 papers as 0). The product already takes the opposite stance one
stage over: "a paper whose rescore call fails is omitted, never scored."

The probe also prices the fix honestly, which is why it does not become a recommendation to
ship a source: an evidence threshold is a **complete no-op on Europe PMC** (100% coverage,
the only currently net-positive source) and moves the OpenAlex arm only -0.76 -> -0.57,
because a display-time cut leaves the displacement term untouched.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import is_arxiv_id  # noqa: E402

RES = EVALS / "results"
WORK = EVALS / ".work"
CONTROL = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json"
ARMS = {
    "epmc": (
        "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json",
        "pool-core25-epmc",
    ),
    "openalex": (
        "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260828T052915Z.json",
        "pool-core25-openalex",
    ),
}
CONTROL_POOL = "pool-core25-arxiv"
THRESHOLDS = (1, 400, 800, 1000)


def pt(score: int) -> int:
    return 1 if int(score) >= 2 else -2


def wilson(k: int, n: int) -> list[float]:
    """95% interval on a proportion. Used because every count here is small."""
    if not n:
        return [0.0, 1.0]
    z, p = 1.96, k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 3), round(centre + half, 3)]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def abstract_lengths(pool: str) -> dict[tuple[str, str], int]:
    out = {}
    for f in sorted((WORK / pool).glob("*.json")):
        for p in json.loads(f.read_text(encoding="utf-8"))["candidates"]:
            out[(f.stem, str(p.get("arxiv_id") or ""))] = len((p.get("abstract") or "").strip())
    return out


def shown(run: list[dict]) -> list[tuple[str, str, int, Any, Any]]:
    """(case, id, judge score, gate score, finescale P) for every paper in a digest."""
    return [
        (
            e["case"],
            str(p["arxiv_id"]),
            int(p["judge_score"]),
            p.get("llm_score"),
            p.get("finescale_p"),
        )
        for e in run
        for p in e["returned"]["reporadar_toppicks"]
    ]


control_run = json.loads((RES / CONTROL).read_text(encoding="utf-8"))
control = shown(control_run)
# Cases come from the RUN, not from the shown papers: a case where every arm abstained shows
# no papers and would vanish from the denominator, quietly turning 37 cases into 36 and
# every per-case mean into a different quantity. `cli`, `http`, `linter` and `webdev` are
# exactly such cases in the control, and they carry 70% of the P26 margin.
cases = sorted({e["case"] for e in control_run})
n_cases = len(cases)
e0 = control_run[0]

out: dict[str, Any] = {
    "_comment": (
        "NR-42: a $0 probe on whether a relevance filter for non-arXiv material is worth "
        "building. Derived by evals/probe_nonarxiv_evidence.py from the three same-day source "
        "arms under evals/results/ and the frozen pools under evals/.work/, both gitignored; "
        "pinned by tests/test_nonarxiv_evidence.py. No LLM or judge calls."
    ),
    "config": {
        "digest_window": e0["digest_window"],
        "gate_depth": e0.get("gate_depth"),
        "n_cases": n_cases,
        "control_run": CONTROL,
    },
    "window_is_not_binding": {},
    "abstract_coverage_in_pool": {},
    "abstract_coverage_when_shown": {},
    "instruments": {},
    "counterfactuals": {},
    "displacement_split": {},
    "oracle_ceiling": {},
}

# ── 1. is the digest window binding? if not, displacement is not slot competition here ──
arm_shown = {"control": control}
for arm, (rf, _pool) in ARMS.items():
    arm_shown[arm] = shown(json.loads((RES / rf).read_text(encoding="utf-8")))
for arm, rows in arm_shown.items():
    per_case: dict[str, int] = {}
    for c, *_ in rows:
        per_case[c] = per_case.get(c, 0) + 1
    sizes = [per_case.get(c, 0) for c in cases]
    out["window_is_not_binding"][arm] = {
        "mean_digest": round(mean(sizes), 1),
        "max_digest": max(sizes),
        "cases_at_the_cap": sum(1 for s in sizes if s >= e0["digest_window"]),
        "n_cases": n_cases,
    }

# ── 2. abstract coverage, in the pool and among the papers actually shown ──
for label, pool in [("control", CONTROL_POOL)] + [(a, p) for a, (_r, p) in ARMS.items()]:
    lens = abstract_lengths(pool)
    ax = [v for (_c, pid), v in lens.items() if is_arxiv_id(pid)]
    na = [v for (_c, pid), v in lens.items() if not is_arxiv_id(pid)]
    out["abstract_coverage_in_pool"][label] = {
        "pool_dir": pool,
        "arxiv_candidates": len(ax),
        "arxiv_with_abstract": round(sum(1 for v in ax if v) / len(ax), 3) if ax else None,
        "arxiv_mean_chars": round(mean(ax)),
        "non_arxiv_candidates": len(na),
        "non_arxiv_with_abstract": round(sum(1 for v in na if v) / len(na), 3) if na else None,
        "non_arxiv_mean_chars": round(mean(na)),
    }

for arm, (_rf, pool) in ARMS.items():
    lens = abstract_lengths(pool)
    na = [(c, pid, j, g, f) for c, pid, j, g, f in arm_shown[arm] if not is_arxiv_id(pid)]
    good = [(c, pid) for c, pid, j, _g, _f in na if j >= 2]
    bad = [(c, pid) for c, pid, j, _g, _f in na if j < 2]
    block = {}
    for name, group in (("actionable", good), ("non_actionable", bad)):
        miss = sum(1 for k in group if not lens.get(k, 0))
        block[name] = {
            "n": len(group),
            "no_abstract": miss,
            "no_abstract_rate": round(miss / len(group), 3) if group else None,
            "ci95": wilson(miss, len(group)),
            "mean_chars": round(mean(lens.get(k, 0) for k in group)),
        }
    out["abstract_coverage_when_shown"][arm] = block

# ── 3. the two instruments the system already has ──
# Both are the stages that solved this problem for arXiv papers (the gate, and the fine-scale
# rescore of the band sitting on it). Neither separates non-arXiv actionable from not.
for arm in ARMS:
    na = [(j, g, f) for _c, pid, j, g, f in arm_shown[arm] if not is_arxiv_id(pid)]
    ax = [(j, g, f) for _c, pid, j, g, f in arm_shown[arm] if is_arxiv_id(pid)]
    good = [(g, f) for j, g, f in na if j >= 2]
    bad = [(g, f) for j, g, f in na if j < 2]
    g3, b3 = sum(1 for g, _ in good if g == 3), sum(1 for g, _ in bad if g == 3)
    band_na = [(j, f) for j, g, f in na if g == 2]
    band_ax = [(j, f) for j, g, f in ax if g == 2]
    out["instruments"][arm] = {
        "gate": {
            "actionable_n": len(good),
            "actionable_gate3": g3,
            "actionable_gate3_rate": round(g3 / len(good), 3) if good else None,
            "actionable_ci95": wilson(g3, len(good)),
            "non_actionable_n": len(bad),
            "non_actionable_gate3": b3,
            "non_actionable_gate3_rate": round(b3 / len(bad), 3) if bad else None,
            "non_actionable_ci95": wilson(b3, len(bad)),
        },
        "rescore": {
            # The rescore keys on `paper["arxiv_id"]`, which non-arXiv papers fill with their
            # DOI, so nothing excludes them -- and nothing does: every band paper carries a
            # `finescale_p`. What it does NOT do is rank the bad ones lower.
            "non_arxiv_band": len(band_na),
            "non_arxiv_band_scored": sum(1 for _j, f in band_na if f is not None),
            "arxiv_band": len(band_ax),
            "arxiv_band_scored": sum(1 for _j, f in band_ax if f is not None),
            "mean_p_actionable": round(mean(f for j, f in band_na if f is not None and j >= 2), 3),
            "mean_p_non_actionable": (
                round(mean(f for j, f in band_na if f is not None and j < 2), 3)
                if any(f is not None and j < 2 for j, f in band_na)
                else None
            ),
            "mean_p_arxiv": round(mean(f for _j, f in band_ax if f is not None), 3),
            "_caveat": (
                "These are papers the rescore ADMITTED, so the distribution is truncated at "
                "its own threshold. It bounds how well the rescore orders within the admitted "
                "set; it does not measure the signal it carries over the papers it rejected, "
                "which this artifact cannot see."
            ),
        },
    }

# ── 4. what the available filters would actually buy ──
for arm, (_rf, pool) in ARMS.items():
    lens = abstract_lengths(pool)
    na = [(c, pid, j, g) for c, pid, j, g, _f in arm_shown[arm] if not is_arxiv_id(pid)]
    source_term = sum(pt(j) for _c, _p, j, _g in na)
    measured = round(
        mean(
            sum(pt(j) for c2, _p, j, _g, _f in arm_shown[arm] if c2 == c)
            - sum(pt(j) for c2, _p, j, _g, _f in control if c2 == c)
            for c in cases
        ),
        2,
    )
    gate3 = [(c, p, j, g) for c, p, j, g in na if g == 3]
    sweep = {}
    for thr in THRESHOLDS:
        keep = [(c, p, j, g) for c, p, j, g in na if lens.get((c, p), 0) >= thr]
        drop = [(c, p, j, g) for c, p, j, g in na if lens.get((c, p), 0) < thr]
        kept_term = sum(pt(j) for _c, _p, j, _g in keep)
        sweep[str(thr)] = {
            "kept": len(keep),
            "dropped_actionable": sum(1 for _c, _p, j, _g in drop if j >= 2),
            "dropped_non_actionable": sum(1 for _c, _p, j, _g in drop if j < 2),
            "source_term": round(kept_term / n_cases, 2),
            "arm_if_slots_unchanged": round(measured + (kept_term - source_term) / n_cases, 2),
        }
    out["counterfactuals"][arm] = {
        "arm_as_measured": measured,
        "source_term": round(source_term / n_cases, 2),
        "gate3_only_filter": {
            "kept": len(gate3),
            "kept_actionable": sum(1 for _c, _p, j, _g in gate3 if j >= 2),
            "dropped_actionable": sum(1 for _c, _p, j, _g in na if j >= 2)
            - sum(1 for _c, _p, j, _g in gate3 if j >= 2),
            "dropped_non_actionable": sum(1 for _c, _p, j, _g in na if j < 2)
            - sum(1 for _c, _p, j, _g in gate3 if j < 2),
            "source_term": round(sum(pt(j) for _c, _p, j, _g in gate3) / n_cases, 2),
        },
        "evidence_threshold_sweep_chars": sweep,
    }

# ── 5. where the cost actually is, and the ceiling on removing it ──
ctrl_by_case: dict[str, list[tuple[str, int]]] = {c: [] for c in cases}
for c, pid, j, _g, _f in control:
    ctrl_by_case[c].append((pid, j))
for arm in ARMS:
    by_case: dict[str, list[tuple[str, int]]] = {c: [] for c in cases}
    for c, pid, j, _g, _f in arm_shown[arm]:
        by_case[c].append((pid, j))
    dropped, added = [], []
    for c in cases:
        base_ids = {p for p, _ in ctrl_by_case[c]}
        treat_ids = {p for p, _ in by_case[c]}
        dropped += [j for p, j in ctrl_by_case[c] if p not in treat_ids]
        added += [j for p, j in by_case[c] if p not in base_ids and is_arxiv_id(p)]
    swap = sum(pt(j) for j in added) - sum(pt(j) for j in dropped)
    rate_out = sum(pt(j) for j in dropped) / len(dropped)
    slots = len(added) - len(dropped)
    out["displacement_split"][arm] = {
        "swap_total": round(swap / n_cases, 2),
        "arxiv_dropped": len(dropped),
        "arxiv_added": len(added),
        "net_slots": slots,
        "value_per_dropped_paper": round(rate_out, 2),
        "slots_term": round(slots * rate_out / n_cases, 2),
        "quality_term": round((swap - slots * rate_out) / n_cases, 2),
    }
    keep = sum(1 for _c, pid, j, _g, _f in arm_shown[arm] if not is_arxiv_id(pid) and j >= 2)
    ctrl_mean = mean(sum(pt(j) for _p, j in ctrl_by_case[c]) for c in cases)
    out["oracle_ceiling"][arm] = {
        "_comment": (
            "A strict over-estimate: every actionable non-arXiv paper added to the control "
            "for free, with zero displacement. No real filter can beat this."
        ),
        "control_mean": round(ctrl_mean, 2),
        "actionable_non_arxiv": keep,
        "ceiling": round(ctrl_mean + keep / n_cases, 2),
        "headroom_over_control": round(keep / n_cases, 2),
    }

(EVALS / "nonarxiv_evidence.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

print(f"window {e0['digest_window']}, gate_depth {e0.get('gate_depth')}, {n_cases} cases\n")
print("digest sizes (the window is not what binds):")
for arm, v in out["window_is_not_binding"].items():
    print(
        f"  {arm:<9} mean {v['mean_digest']:>4}  at the cap {v['cases_at_the_cap']}/{v['n_cases']}"
    )
print("\nabstract coverage in the pool:")
for arm, v in out["abstract_coverage_in_pool"].items():
    print(
        f"  {arm:<9} arXiv {v['arxiv_with_abstract']} ({v['arxiv_mean_chars']} ch)   "
        f"non-arXiv {v['non_arxiv_with_abstract']} ({v['non_arxiv_mean_chars']} ch, "
        f"n={v['non_arxiv_candidates']})"
    )
print("\nabstract coverage among papers SHOWN:")
for arm, v in out["abstract_coverage_when_shown"].items():
    for name, b in v.items():
        print(
            f"  {arm:<9} {name:<15} {b['no_abstract']}/{b['n']} have none "
            f"({b['no_abstract_rate']}) CI {b['ci95']}  mean {b['mean_chars']} ch"
        )
print("\ninstruments (neither separates):")
for arm, v in out["instruments"].items():
    g = v["gate"]
    r = v["rescore"]
    print(
        f"  {arm:<9} gate-3 rate  actionable {g['actionable_gate3_rate']} "
        f"vs non-actionable {g['non_actionable_gate3_rate']}"
    )
    print(
        f"  {'':<9} rescore P    actionable {r['mean_p_actionable']} "
        f"vs non-actionable {r['mean_p_non_actionable']}   "
        f"(band scored {r['non_arxiv_band_scored']}/{r['non_arxiv_band']})"
    )
print("\ncounterfactuals:")
for arm, v in out["counterfactuals"].items():
    g = v["gate3_only_filter"]
    print(
        f"  {arm:<9} source term {v['source_term']:+.2f} -> gate-3 only {g['source_term']:+.2f} "
        f"(loses {g['dropped_actionable']} actionable to remove {g['dropped_non_actionable']})"
    )
    for thr, s in v["evidence_threshold_sweep_chars"].items():
        print(
            f"  {'':<9}   >= {thr:>4} ch: source {s['source_term']:+.2f}  "
            f"arm {v['arm_as_measured']:+.2f} -> {s['arm_if_slots_unchanged']:+.2f}"
        )
print("\ndisplacement and ceiling:")
for arm in ARMS:
    d, o = out["displacement_split"][arm], out["oracle_ceiling"][arm]
    print(
        f"  {arm:<9} swap {d['swap_total']:+.2f} = slots {d['slots_term']:+.2f} "
        f"+ quality {d['quality_term']:+.2f}   |   oracle ceiling {o['ceiling']:+.2f} "
        f"(headroom {o['headroom_over_control']:+.2f})"
    )
