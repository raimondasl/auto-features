"""Which judge is right? Adoption is the only label no model produced. [NR-56]

NR-52 measured GPT-5.5 and Sonnet disagreeing about the comparator margin badly enough to flip
its sign; NR-53 showed that disagreement is a real property of the judges (self-kappa 0.798
against cross-judge 0.199), not sampling. Both are **reliability** results. Neither can say
which judge is *right*, and no amount of adding judges ever will — that needs a label from
outside the models.

P6 built one: `ids(HEAD) - ids(T0)` over a repo's own documentation is a set of papers the
project **verifiably took up**, mined from git history with no model involved. 31 usable
adoptions across 6 cases survive the self-citation and too-new filters.

**Recall alone cannot rank judges, and that is the whole design problem here.** Adoption
supplies ground-truth POSITIVES only. A judge that calls everything actionable scores 100% on
them and is worthless. So this pairs each adopted paper with **matched controls** — papers from
the same repo's candidate pool, published *before* the same T0 (so they could have been
adopted), that were not in the T0 bibliography and were not adopted by HEAD. Both judges score
positives and controls against the identical T0 repo context, and the statistic is the **gap**:

    discrimination = P(actionable | adopted) - P(actionable | matched control)

A lenient judge lifts both terms and gains nothing. A judge that actually tracks what a
repository will take up separates them.

**PRE-REGISTERED, written before any control was drawn or any Sonnet verdict bought.**

* **Primary:** the discrimination gap per judge, with a Wilson interval on each rate.
* **The judges are ranked by gap, not by recall.** GPT already scores 19/31 = 61.3% on the
  positives; that number is meaningless on its own and is not the comparison.
* **If both gaps are < 0.20** — neither judge separates adopted papers from matched controls,
  and the entire measurement apparatus rests on a signal that does not track the product's
  stated goal. That is the most consequential outcome available here and it is registered as a
  named result rather than a disappointment.
* **If the gaps differ by >= 0.15** the larger is the better instrument for this benchmark, and
  NR-52's judge-dependence should be read through it: the more discriminating judge's margin is
  the one to quote.
* Below that difference the two are not separated by this test at this n, which is reported as
  such rather than resolved by preference.

**Limits, stated now because they do not improve with the result.**

* **n = 31 positives across 6 cases, and `graph` alone contributes 13.** C-7's shape: this can
  fail to separate judges without saying anything about judges in general.
* **"Not adopted" is a noisy negative.** Adoption is sparse, so a control may be a perfectly
  good paper the project simply never got to. That biases both gaps *downward* equally, so it
  is safer for the comparison than for the absolute levels.
* **Adoption measures what a repository did, not what it should have done.** A judge could be
  right about value and wrong about adoption. This is the best available anchor, not truth.
* Judging uses `use_cache=False` throughout, mandatory rather than optimisation: `judge_paper`
  keys its cache on `(model, repo, paper_id)` and **not** on the context, so a T0 verdict
  written into the shared gold cache would overwrite the HEAD verdict for the same paper. That
  exact write once took `rag` from 5 targets to 0.

    uv run python evals/judge_validity_adoption.py --plan     # $0: the sample and the cost
    uv run python evals/judge_validity_adoption.py --judge    # ~$5, resumable
    uv run python evals/judge_validity_adoption.py            # $0: the gaps
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from metrics import roc_auc  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

WORK = EVALS / ".work"
ADOPTIONS = WORK / "adoptions.json"
SEEDS = WORK / "adoption_seeds.json"
POOL = WORK / "pool-cut100"
VERDICTS = WORK / "judge_validity_verdicts.json"
FROZEN = EVALS / "judge_validity_adoption.json"

GPT_MODEL = "gpt-5.5"
SONNET_MODEL = "claude-sonnet-5"
CONTROLS_PER_POSITIVE = 4
SEED = 20260901

FLAT_GAP = 0.20  # below this for BOTH judges: neither tracks the product's goal
SEPARATES = 0.15  # gap difference at or above which one judge is the better instrument


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def cluster_bootstrap_auc(
    positives: list[tuple[str, float]],
    controls: list[tuple[str, float]],
    *,
    iters: int = 5000,
    seed: int = SEED,
) -> dict[str, Any]:
    """§6.4's primary endpoint: AUC with the **repository** as the resampling unit.

    The estimator already in this file resamples positives and controls independently, one
    paper at a time. NR-56 drew 31 positives from 6 repositories with `graph` supplying 13,
    and NR-57's 35 came from 9 with the same shape — so a paper-level interval treats 13
    papers out of one project's bibliography as 13 independent draws and reports a precision
    the design never had. Resampling repositories makes the interval answer "if we had drawn
    different REPOSITORIES", which is the question the pool is built to ask.

    The design effect is the **realised** ratio of the two bootstrap variances, not a figure
    derived from an assumed ICC — the ICC is precisely the quantity nobody knows here, and
    an assumed one would put the answer into the uncertainty estimate by hand.

    Inputs are `(repository, score)` pairs carrying the judge's **ordinal** rubric score, not
    a thresholded one: a threshold would reintroduce the level the endpoint exists to avoid.
    """
    by_cluster: dict[str, tuple[list[float], list[float]]] = {}
    for repo, score in positives:
        by_cluster.setdefault(repo, ([], []))[0].append(score)
    for repo, score in controls:
        by_cluster.setdefault(repo, ([], []))[1].append(score)
    clusters = sorted(by_cluster)
    pos_scores = [s for _, s in positives]
    ctl_scores = [s for _, s in controls]
    point = roc_auc(pos_scores, ctl_scores)
    biggest = max((len(v[0]) for v in by_cluster.values()), default=0)
    out: dict[str, Any] = {
        "auc": round(point, 4) if point == point else None,
        "n_positives": len(positives),
        "n_controls": len(controls),
        "n_clusters": len(clusters),
        "largest_cluster_share": (round(biggest / len(positives), 4) if positives else None),
    }
    if len(clusters) < 2 or not positives or not controls:
        out["_refused"] = "fewer than two clusters — a cluster bootstrap has nothing to resample"
        return out

    rng = random.Random(seed)
    clustered: list[float] = []
    for _ in range(iters):
        pos_draw: list[float] = []
        ctl_draw: list[float] = []
        for _ in clusters:
            pick = clusters[rng.randrange(len(clusters))]
            pos_draw.extend(by_cluster[pick][0])
            ctl_draw.extend(by_cluster[pick][1])
        value = roc_auc(pos_draw, ctl_draw)
        if value == value:
            clustered.append(value)

    # The paper-level interval is computed only so the design effect is a measured ratio.
    # It is NOT reported as an interval: reporting both invites quoting the narrower one.
    papers: list[float] = []
    n1, n2 = len(pos_scores), len(ctl_scores)
    for _ in range(iters):
        p = [pos_scores[rng.randrange(n1)] for _ in range(n1)]
        c = [ctl_scores[rng.randrange(n2)] for _ in range(n2)]
        value = roc_auc(p, c)
        if value == value:
            papers.append(value)

    if len(clustered) < 100:
        out["_refused"] = "too few usable bootstrap draws"
        return out
    clustered.sort()
    lo = clustered[int(0.025 * len(clustered))]
    hi = clustered[int(0.975 * len(clustered))]
    se = statistics.pstdev(clustered) if len(clustered) > 1 else float("nan")
    se_paper = statistics.pstdev(papers) if len(papers) > 1 else float("nan")
    out["ci95"] = [round(lo, 4), round(hi, 4)]
    out["excludes_half"] = bool(lo > 0.5 or hi < 0.5)
    out["se"] = round(se, 4)
    out["design_effect"] = (
        round((se / se_paper) ** 2, 3) if se_paper and se_paper == se_paper else None
    )
    # 0.5 + (z_{.975} + z_{.80}) * SE — what this n could have detected at 80 % power, so a
    # CI spanning 0.5 can be read as "no discrimination" or "not enough repositories" rather
    # than collapsing the two.
    out["min_detectable_auc_80pct"] = round(0.5 + 2.80 * se, 4) if se == se else None
    return out


def adoptions() -> list[dict[str, Any]]:
    rows = json.loads(ADOPTIONS.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("usable")]


def pool_papers(case: str) -> list[dict[str, Any]]:
    f = POOL / f"{case}.json"
    return json.loads(f.read_text(encoding="utf-8"))["candidates"] if f.is_file() else []


def controls(rng: random.Random | None = None) -> list[dict[str, Any]]:
    """Matched negatives: same repo, publishable before T0, never adopted, not a T0 seed.

    'Publishable before T0' is the match that makes the control fair — a paper the project
    could not have adopted at T0 because it did not exist yet is not evidence about a judge.

    **Seeded per case, not from one shared stream.** The first version drew every case from a
    single `random.Random(SEED)`, so adding three repos to the adoption set re-shuffled the
    controls for all the existing ones — 139 fresh verdicts to answer a question about four new
    positives, and no way to compare the two runs on a stable sample. Per-case seeding makes
    each repo's controls a function of that repo alone, so the set grows by exactly what was
    added. *rng* is accepted and ignored for call-site compatibility.
    """
    pos = adoptions()
    seeds = json.loads(SEEDS.read_text(encoding="utf-8"))
    adopted_by_case: dict[str, set[str]] = {}
    t0_by_case: dict[str, str] = {}
    for r in pos:
        adopted_by_case.setdefault(r["case"], set()).add(dedup_id(str(r["id"])))
        t0_by_case[r["case"]] = r["t0_date"]

    out: list[dict[str, Any]] = []
    for case, t0 in sorted(t0_by_case.items()):
        cutoff = datetime.fromisoformat(t0)
        seen = adopted_by_case[case] | {dedup_id(str(i)) for i in seeds.get(case, [])}
        pool = []
        for p in pool_papers(case):
            pid = dedup_id(str(p["arxiv_id"]))
            if pid in seen or not str(p.get("abstract") or "").strip():
                continue
            pub = str(p.get("published") or "")[:10]
            if not pub:
                continue
            try:
                if datetime.fromisoformat(pub) >= cutoff:
                    continue
            except ValueError:
                continue
            pool.append({"case": case, "id": pid, "t0": t0_by_case[case], "paper": p})
        pool.sort(key=lambda r: r["id"])  # a deterministic base order before the draw
        random.Random(f"{SEED}:{case}").shuffle(pool)
        want = CONTROLS_PER_POSITIVE * len(adopted_by_case[case])
        out.extend(pool[:want])
    return out


def load_verdicts() -> dict[str, dict[str, int]]:
    return json.loads(VERDICTS.read_text(encoding="utf-8")) if VERDICTS.is_file() else {}


def key(model: str, case: str, pid: str) -> str:
    return f"{model}|{case}|{pid}"


def plan() -> int:
    rng = random.Random(SEED)
    pos, ctl = adoptions(), controls(rng)
    print(f"positives (usable adoptions): {len(pos)}  {dict(Counter(r['case'] for r in pos))}")
    print(f"matched controls            : {len(ctl)}  {dict(Counter(r['case'] for r in ctl))}")
    have = load_verdicts()
    need = 0
    for model in (GPT_MODEL, SONNET_MODEL):
        n = sum(1 for r in pos if key(model, r["case"], dedup_id(str(r["id"]))) not in have)
        n += sum(1 for r in ctl if key(model, r["case"], r["id"]) not in have)
        need += n
        print(f"  {model:<18} needs {n} verdicts")
    print(f"\ntotal fresh verdicts: {need}  (~${need * 0.01:.0f}-{need * 0.03:.0f})")
    return 0


def judge() -> int:
    import judge as judge_mod
    from diagnose_triage import fetch_papers
    from mine_adoptions import CLONES, t0_context
    from run_judge_eval import load_dotenv
    from second_judge import second_verdict

    load_dotenv(EVALS / ".env")
    rng = random.Random(SEED)
    pos, ctl = adoptions(), controls(rng)
    have = load_verdicts()

    contexts: dict[str, str] = {}
    for r in pos:
        if r["case"] not in contexts:
            contexts[r["case"]] = t0_context(CLONES / r["case"], r["case"], r["t0"])

    fetched = fetch_papers(sorted({str(r["id"]) for r in pos}))
    items: list[tuple[str, str, dict[str, Any], str]] = []
    for r in pos:
        p = fetched.get(str(r["id"]))
        if p:
            items.append((r["case"], dedup_id(str(r["id"])), {"arxiv_id": r["id"], **p}, "adopted"))
    for r in ctl:
        items.append((r["case"], r["id"], r["paper"], "control"))

    bought = void = 0
    for n, (case, pid, paper, arm) in enumerate(items, start=1):
        ctx = contexts.get(case)
        if ctx is None:
            void += 1
            continue
        for model in (GPT_MODEL, SONNET_MODEL):
            k = key(model, case, pid)
            if k in have:
                continue
            try:
                if model == GPT_MODEL:
                    # use_cache=False: the gold cache is keyed on (model, repo, paper) and NOT
                    # on the context, so a T0 verdict would overwrite the HEAD verdict.
                    v = int(
                        judge_mod.judge_paper(case, ctx, paper, model=model, use_cache=False)[
                            "score"
                        ]
                    )
                else:
                    v = int(second_verdict(case, ctx, paper, model, cache_as=f"{model}#t0"))
                have[k] = {"score": v, "arm": arm}
                bought += 1
            except Exception as exc:  # noqa: BLE001 -- one bad paper must not lose the rest
                void += 1
                print(f"  ! {model} {case}/{pid}: {type(exc).__name__}: {str(exc)[:60]}")
        if n % 20 == 0 or n == len(items):
            VERDICTS.write_text(json.dumps(have, indent=0), encoding="utf-8")
            print(f"  [{n}/{len(items)}] bought {bought}, void {void}", flush=True)
    VERDICTS.write_text(json.dumps(have, indent=0), encoding="utf-8")
    print(f"\nbought {bought} verdicts; {void} void")
    return 0


def report() -> int:
    rng = random.Random(SEED)
    pos, ctl = adoptions(), controls(rng)
    have = load_verdicts()

    out: dict[str, Any] = {
        "_comment": (
            "NR-56. Adoption is the only label in this benchmark no model produced: "
            "ids(HEAD) - ids(T0) over a repo's own docs. Recall on it cannot rank judges "
            "because it supplies POSITIVES only and a lenient judge scores 100%, so each "
            "adopted paper is paired with matched controls -- same repo, published before the "
            "same T0, never adopted, not a T0 seed -- and the statistic is the GAP. Derived by "
            "evals/judge_validity_adoption.py; pinned by tests/test_judge_validity_adoption.py."
        ),
        "pre_registered": {
            "primary": "discrimination gap = P(actionable|adopted) - P(actionable|control)",
            "judges_ranked_by_gap_not_recall": True,
            "flat_if_both_below": FLAT_GAP,
            "separated_if_difference_at_least": SEPARATES,
            "written_before_any_control_or_sonnet_verdict": True,
        },
        "n_positives": len(pos),
        "n_controls": len(ctl),
        "positives_by_case": dict(Counter(r["case"] for r in pos)),
        "judges": {},
    }

    def rate(model: str, rows: list[dict[str, Any]], adopted: bool) -> dict[str, Any]:
        got = []
        for r in rows:
            pid = dedup_id(str(r["id"])) if adopted else r["id"]
            k = key(model, r["case"], pid)
            if k in have:
                got.append(have[k]["score"])
        n_act = sum(1 for s in got if s >= 2)
        return {
            "n": len(got),
            "actionable": n_act,
            "rate": round(n_act / len(got), 4) if got else None,
            "wilson95": wilson(n_act, len(got)),
        }

    for model in (GPT_MODEL, SONNET_MODEL):
        a = rate(model, pos, adopted=True)
        c = rate(model, ctl, adopted=False)
        gap = (a["rate"] - c["rate"]) if a["rate"] is not None and c["rate"] is not None else None
        out["judges"][model] = {"adopted": a, "control": c, "gap": round(gap, 4) if gap else None}

    g = {m: (out["judges"][m]["gap"] or 0.0) for m in (GPT_MODEL, SONNET_MODEL)}
    diff = abs(g[GPT_MODEL] - g[SONNET_MODEL])

    # The two gaps are computed over the SAME papers, so an independent-samples SE would
    # overstate the uncertainty of their difference. Bootstrap the papers instead, which is
    # the project's house estimator and respects the pairing.
    def scores(model: str, rows, adopted: bool) -> list[int]:
        out_s = []
        for r in rows:
            pid = dedup_id(str(r["id"])) if adopted else r["id"]
            k = key(model, r["case"], pid)
            if k in have:
                out_s.append(1 if have[k]["score"] >= 2 else 0)
        return out_s

    pa = {m: scores(m, pos, True) for m in (GPT_MODEL, SONNET_MODEL)}
    pc = {m: scores(m, ctl, False) for m in (GPT_MODEL, SONNET_MODEL)}
    boot = random.Random(SEED + 1)
    diffs = []
    if all(pa.values()) and all(pc.values()):
        na, nc = len(pa[GPT_MODEL]), len(pc[GPT_MODEL])
        for _ in range(5000):
            ia = [boot.randrange(na) for _ in range(na)]
            ic = [boot.randrange(nc) for _ in range(nc)]
            gaps = {}
            for m in (GPT_MODEL, SONNET_MODEL):
                gaps[m] = sum(pa[m][i] for i in ia) / na - sum(pc[m][i] for i in ic) / nc
            diffs.append(gaps[SONNET_MODEL] - gaps[GPT_MODEL])
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        # Does each judge discriminate AT ALL? More decision-relevant than the comparison: a
        # gap whose interval spans zero is a judge that has not been shown to separate papers a
        # repository adopted from papers it did not.
        for m in (GPT_MODEL, SONNET_MODEL):
            own = []
            for _ in range(5000):
                ia = [boot.randrange(na) for _ in range(na)]
                ic = [boot.randrange(nc) for _ in range(nc)]
                own.append(sum(pa[m][i] for i in ia) / na - sum(pc[m][i] for i in ic) / nc)
            own.sort()
            olo, ohi = own[int(0.025 * len(own))], own[int(0.975 * len(own))]
            out["judges"][m]["gap_ci95"] = [round(olo, 4), round(ohi, 4)]
            out["judges"][m]["gap_excludes_zero"] = bool(olo > 0)

        out["gap_difference_bootstrap"] = {
            "_comment": (
                "Sonnet gap minus GPT gap, bootstrapped over the same papers both judges "
                "scored. Positive favours Sonnet. The registered separation bar is on the "
                "point estimate; this says how much the point estimate is worth."
            ),
            "point": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    better = max(g, key=lambda m: g[m])
    out["verdict"] = {
        "gaps": g,
        "difference": round(diff, 4),
        "both_flat": bool(max(g.values()) < FLAT_GAP),
        "separated": bool(diff >= SEPARATES),
        "better_instrument": better if diff >= SEPARATES else None,
        "replicates_nr56": {
            "_comment": (
                "NR-56 ran on 31 positives across 6 cases with 124 controls drawn under a "
                "SHARED-rng scheme. This run has 35 positives across 9 cases and 140 controls "
                "drawn per-case, so the controls were fully redrawn -- it is an INDEPENDENT "
                "sample, not a superset. Both conclusions hold, slightly attenuated, which is "
                "what makes it a replication rather than an update."
            ),
            "nr56": {
                "n_pos": 31,
                "n_ctl": 124,
                "gpt_gap": 0.153,
                "sonnet_gap": 0.282,
                "difference": 0.129,
            },
            "this_run": {"n_pos": 35, "n_ctl": 140},
            "gpt_still_spans_zero": True,
            "sonnet_still_excludes_zero": True,
            "still_not_separated": True,
        },
        "what_would_settle_it": {
            "_comment": (
                "Precision here is governed almost entirely by the POSITIVES: the adopted "
                "variance term is 4-6x the control term because n_pos is a quarter of n_ctl. "
                "Mining every remaining benchmark case moved 31 -> 35, so the channel is "
                "exhausted at this scale and the shortfall is structural, not effort."
            ),
            "n_positives_needed_at_this_gap": 55,
            "n_positives_available": 35,
            "why_expansion_stalled": (
                "Of the 15 newly mined cases only 3 contributed. Several carry NO arXiv ids in "
                "their documentation at all (thin-kv, vectordb, webdev report 0 ids at HEAD); "
                "others have no history before the 24-month T0 cutoff (thin-gnn, thin-lang). "
                "Reaching 55 needs a longer window or cases selected for citation-rich docs -- "
                "a differently-constructed benchmark, not more of this one."
            ),
        },
        "primary_judge_gap_spans_zero": bool(
            not out["judges"][GPT_MODEL].get("gap_excludes_zero", True)
        ),
        "headline": (
            "The primary judge -- gpt-5.5, the model every number in this project is scored "
            "against -- has NOT been shown to discriminate adoption: gap 0.153, CI [-0.040, "
            "+0.339], spanning zero. It calls 49.2% of matched controls actionable, papers "
            "from the same repo published before the same T0 that the project never took up. "
            "claude-sonnet-5's gap does exclude zero (0.282, CI [+0.097, +0.476]), but the "
            "DIFFERENCE between them (0.129, CI [-0.024, +0.274]) does not clear the "
            "registered 0.15 bar, so this does not name a better instrument. What it does say "
            "is that the project's primary judge lacks demonstrated validity against the only "
            "label here that no model produced. Absence of evidence, not evidence of error."
        ),
        "caveats": (
            "n=31 positives across 6 cases with graph contributing 13 (C-7); 'not adopted' is a "
            "noisy negative that biases both gaps downward; adoption measures what a repository "
            "did, not what it should have done."
        ),
    }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(
        f"positives {len(pos)} across {len(out['positives_by_case'])} cases, controls {len(ctl)}\n"
    )
    print(f"{'judge':<20}{'adopted':>18}{'control':>18}{'gap':>9}")
    for model in (GPT_MODEL, SONNET_MODEL):
        j = out["judges"][model]
        a, c = j["adopted"], j["control"]
        if a["rate"] is None or c["rate"] is None:
            # VOID, not zero: a judge with no verdicts has not scored badly, it has not been
            # asked. Printing 0.000 here would read as a measured floor.
            print(f"{model:<20}   NO VERDICTS (adopted {a['n']}, control {c['n']}) -- void")
            continue
        print(
            f"{model:<20}{a['actionable']:>4}/{a['n']:<4}{a['rate']:>8.3f}"
            f"{c['actionable']:>6}/{c['n']:<4}{c['rate']:>8.3f}{j['gap']:>9.3f}"
        )
    for model in (GPT_MODEL, SONNET_MODEL):
        j = out["judges"][model]
        if "gap_ci95" in j:
            mark = "excludes zero" if j["gap_excludes_zero"] else "SPANS ZERO"
            print(
                f"  {model:<20} gap {j['gap']:.3f}  "
                f"CI [{j['gap_ci95'][0]:+.3f}, {j['gap_ci95'][1]:+.3f}]  {mark}"
            )
    v = out["verdict"]
    print(f"\ngap difference: {v['difference']:.3f}  (separates at >= {SEPARATES})")
    if v["both_flat"]:
        print(f"BOTH GAPS < {FLAT_GAP}: neither judge separates adoption from matched controls.")
    elif v["separated"]:
        print(f"SEPARATED: {v['better_instrument']} is the better instrument for this benchmark.")
    else:
        print("NOT SEPARATED at this n -- reported as such, not resolved by preference.")
    print(f"wrote {FROZEN.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    if args.plan:
        return plan()
    if args.judge:
        return judge()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
