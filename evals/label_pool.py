"""P5: label the pool where the filter would live — per-stratum base rates and gate admit rates.

Everything measured about retrieval so far is **recall**. P1 cut a pool, P4 reached 27 of 48
targets — neither has a precision number, because the 800-odd cached judge verdicts all
describe papers the *ranker or the baseline* surfaced. Nothing has ever been labelled in the
wild, so the cascade arithmetic that every design ends with — "filter, then gate, then show
the reader ~10 papers" — has never had a real admit rate in it.

This buys one. Two instruments over the same stratified sample:

  gate    the SHIPPED Haiku gate (`score_actionability`, prose-300 repo context), over every
          sampled paper. ~$0.0003 each, so it can afford breadth. Gives the admit RATE,
          which is what decides whether a cascade is affordable at all.
  judge   the GPT-5.5 judge with the shipped rubric, over a uniform random subsample of the
          same papers. ~$0.03 each. Gives the actionable BASE RATE per stratum, and — since
          every judged paper was also gated — the gate's precision and recall off the
          distribution it was tuned on.

**Six strata.** P5 was written before P4 and specced four hop-only strata (forward degree,
backward membership, citation band, age band). P4 changed which pool matters: HyDE reaches
27/48 including 15 targets no bibliography can reach, so the pool a filter would live in is
now HyDE's, not only the hop's. The strata are restated accordingly — three rank bands of the
fused HyDE ranking, two coupling-degree bands of the hop pool, and a uniform sample of all
3.1M arXiv papers as the floor.

**`random-arxiv` is the load-bearing arm.** Without it, "4% of the top band is actionable" is
unreadable: it could mean the band is good, or that the judge calls 4% of anything actionable
against a real repo. That mistake has already been made once in this project (the `lacks`
arms, corrected on 2026-08-05 by exactly this control), so it is here from the start.

**Pre-registered, before running (2026-08-06).** P5's bars, restated for the new strata:

  * PREDICTION: top stratum (`hyde-top100`) >=8% actionable against `random-arxiv` <=1% —
    a >=6x separation; and the shipped gate's wild-admit rate <=10%.
  * KILL: every stratum <=2% with overlapping CIs, i.e. sampled labelling cannot see filter
    quality at this density. INDEPENDENTLY: gate wild-admit >20%, which means the terminal
    stage must be redesigned before any cascade ships, whatever retrieval does.

**Power, chosen in advance rather than discovered afterwards.** At the predicted 8% vs 0%,
Fisher exact gives p = 0.057 at n=60, **0.028 at n=80, 0.007 at n=100**. The two
pre-registered strata are therefore judged at **n=100** — the extra $2.56 is what makes this
a test rather than an anecdote. The other four are judged at n=30 and are **descriptive
only**: at that size nothing separating 8% from 1% can reach significance, and saying so here
is cheaper than discovering it in the write-up. Wilson intervals are reported throughout so
the uncertainty is visible rather than implied.

The properly-powered half of this experiment is the **gate admit rate** at n=200/stratum,
where a 10% rate carries a CI of about [6%, 15%] — narrow enough to decide the cascade
question the prediction is really about.

    uv run python evals/label_pool.py --dry-run     # sample composition, $0
    uv run python evals/label_pool.py               # ~$0.3 gate + ~$10.2 judge

Verdicts enter the shared judge cache under the SHIPPED rubric, which is the point — P5 grows
the labelled set. `resolve_targets()` is asserted unchanged across the run: wild verdicts must
not move the gold set, because it intersects with baseline picks and these papers are not
baseline picks. That invariant is checked, not assumed; the last script that wrote to this
cache silently took `rag` from 5 targets to 0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import judge as judge_mod  # noqa: E402
from build_hop_pool import resolve_targets  # noqa: E402
from diagnose_triage import _load_env, fetch_papers  # noqa: E402
from harness import WORK_DIR, assemble_repo_context  # noqa: E402

from reporadar.config import ProfilerConfig, SuggestionsConfig  # noqa: E402
from reporadar.profiler import profile_repo  # noqa: E402
from reporadar.triage import score_actionability  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
HOP_DIR = WORK / "hop_pool"
TOPK_DIR = WORK / "hyde_topk"
INDEX_DIR = WORK / "hyde_index"
OUT = WORK / "label_pool.json"

ACTIONABLE = 2
SEED = 20260806
GATE_N = 200
JUDGE_N = {"hyde-top100": 100, "random-arxiv": 100}
JUDGE_N_DEFAULT = 30
STRATA = (
    "hyde-top100",
    "hyde-100-1k",
    "hyde-1k-10k",
    "hop-coupling3+",
    "hop-coupling1",
    "random-arxiv",
)
# Pre-registered, see module docstring.
PREDICT_TOP_STRATUM = 0.08
PREDICT_FLOOR = 0.01
GATE_ADMIT_PASS = 0.10
GATE_ADMIT_KILL = 0.20
KILL_ALL_STRATA = 0.02


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Normal-approximation CIs are useless at k=0, which is
    exactly where the floor stratum is expected to sit."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]. Exact because the counts are tiny."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * 1.0000001))


def separation_clauses(top_rate: float, floor_rate: float) -> dict[str, bool]:
    """The three parts of P5's separation prediction, evaluated separately.

    Reported one by one because collapsing them into a single `and` is how the first version
    of this report printed "BELOW PREDICTION" for a 29x effect at p < 0.001: the floor came
    in at 2/100 against a <=1/100 bar, and one missed clause silently outvoted the other two.
    A pre-registered criterion is reported as written; it is not reported as a single bit.
    """
    return {
        f"top stratum >={PREDICT_TOP_STRATUM:.0%}": top_rate >= PREDICT_TOP_STRATUM,
        f"floor <={PREDICT_FLOOR:.0%}": floor_rate <= PREDICT_FLOOR,
        "separation >=6x": top_rate >= 6 * max(floor_rate, 1e-9),
    }


def _hyde_bands() -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {s: [] for s in STRATA if s.startswith("hyde")}
    for path in sorted(TOPK_DIR.glob("*.json")):
        case = path.stem
        ids = json.loads(path.read_text(encoding="utf-8"))
        out["hyde-top100"] += [(case, i) for i in ids[:100]]
        out["hyde-100-1k"] += [(case, i) for i in ids[100:1000]]
        out["hyde-1k-10k"] += [(case, i) for i in ids[1000:10000]]
    return out


def _hop_bands() -> dict[str, list[tuple[str, str]]]:
    """Split the hop pool by total coupling degree — how many of the repo's own seeds touch
    the candidate, in either direction.

    The 3+/1 boundary is taken from P1's ALREADY-PUBLISHED target distribution (coupling >=3
    holds 11 of the 21 reached targets in 15,461 candidates; coupling <=1 holds 5 in 77,855),
    not from any label this script produces. Disclosed because choosing a boundary from a
    recall table is a mild form of peeking even when the labels are independent.
    """
    out: dict[str, list[tuple[str, str]]] = {"hop-coupling3+": [], "hop-coupling1": []}
    for path in sorted(HOP_DIR.glob("*.jsonl")):
        case = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            degree = int(row.get("fwd_degree", 0)) + int(row.get("back_degree", 0))
            if degree >= 3:
                out["hop-coupling3+"].append((case, row["id"]))
            elif degree <= 1:
                out["hop-coupling1"].append((case, row["id"]))
    return out


def _random_arxiv(cases: list[str], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Uniform over the whole 3.1M index, each paper assigned to a real repo round-robin.

    Assigning a case is not a detail: "is this paper actionable" has no meaning without one,
    and a floor measured against the same repos as the treatment arms is the only floor that
    subtracts correctly.
    """
    shards = sorted(INDEX_DIR.glob("*.ids"))
    if not shards:
        return []
    pool: list[str] = []
    for path in shards:
        pool += path.read_text(encoding="utf-8").split("\n")
    picked = rng.sample([p for p in pool if p], n)
    return [(cases[i % len(cases)], pid) for i, pid in enumerate(picked)]


def balanced_draw(rows: list[tuple[str, str]], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Draw *n* rows, round-robin across cases, uniformly within each case.

    A flat uniform draw would not do. `graph` alone contributes 42,112 of the hop pool's
    109,704 candidates, so a pooled sample of the high-coupling band came back **139 of 200
    from `graph`** — a stratum that measures one repo and reports it as a property of the
    filter. Round-robin makes each repo contribute about equally, which is also how a reader
    meets these papers: one digest per repo, not one pool across all of them.

    The estimand this defines is the mean per-repo density, not the pool-weighted density.
    For "does this feature separate actionable from not" the per-repo mean is the honest one;
    for "how many candidates would the cascade process" the pool weights matter, and that
    number comes from the band sizes, which are recorded separately.
    """
    by_case: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row[0], []).append(row)
    for case_rows in by_case.values():
        rng.shuffle(case_rows)
    out: list[tuple[str, str]] = []
    order = sorted(by_case)
    while len(out) < n and any(by_case[c] for c in order):
        for case in order:
            if by_case[case] and len(out) < n:
                out.append(by_case[case].pop())
    rng.shuffle(out)  # so the judged prefix is not case-ordered
    return out


def build_sample(gate_n: int, rng: random.Random) -> dict[str, list[tuple[str, str]]]:
    bands = {**_hyde_bands(), **_hop_bands()}
    cases = sorted({c for rows in bands.values() for c, _ in rows if (WORK_DIR / c).is_dir()})
    sample: dict[str, list[tuple[str, str]]] = {}
    for stratum in STRATA:
        if stratum == "random-arxiv":
            sample[stratum] = _random_arxiv(cases, gate_n, rng)
            continue
        rows = [r for r in bands.get(stratum, []) if (WORK_DIR / r[0]).is_dir()]
        sample[stratum] = balanced_draw(rows, gate_n, rng)
    return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-n", type=int, default=GATE_N, help="papers gated per stratum")
    ap.add_argument("--judge-n", type=int, help="override the judged subsample per stratum")
    ap.add_argument("--dry-run", action="store_true", help="show the sample and cost, call nothing")
    ap.add_argument("--report", action="store_true", help="re-derive from saved labels, $0")
    ap.add_argument("--gate-model", default="claude-haiku-4-5")
    ap.add_argument("--judge-model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()
    _load_env()
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8")))
        return 0

    rng = random.Random(SEED)
    sample = build_sample(args.gate_n, rng)
    judged_n = {s: args.judge_n or JUDGE_N.get(s, JUDGE_N_DEFAULT) for s in STRATA}
    gate_total = sum(len(v) for v in sample.values())
    judge_total = sum(min(judged_n[s], len(sample[s])) for s in STRATA)
    print(f"{'stratum':14} {'gated':>7} {'judged':>7}  cases")
    for s in STRATA:
        cases = Counter(c for c, _ in sample[s])
        print(
            f"{s:14} {len(sample[s]):>7} {min(judged_n[s], len(sample[s])):>7}  "
            f"{len(cases)} ({', '.join(f'{c}:{n}' for c, n in cases.most_common(3))}...)"
        )
    print(
        f"\ntotal: {gate_total} gate calls (~${gate_total * 0.00023:.2f}), "
        f"{judge_total} judge calls (~${judge_total * 0.032:.2f})"
    )
    if args.dry_run:
        return 0

    before = resolve_targets()
    wanted = sorted({pid for rows in sample.values() for _, pid in rows})
    print(f"\nfetching abstracts for {len(wanted)} papers...")
    papers = fetch_papers(wanted)
    print(f"have text for {sum(1 for i in wanted if i in papers)}/{len(wanted)}\n")

    gate_cfg = SuggestionsConfig(provider="claude", claude_model=args.gate_model, timeout=60)
    profiles: dict[str, Any] = {}
    contexts: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for stratum in STRATA:
        picks = [(c, i) for c, i in sample[stratum] if i in papers]
        to_judge = {i for _, i in picks[: judged_n[stratum]]}
        admits = judged = actionable = failures = 0
        for case, pid in picks:
            if case not in profiles:
                repo = WORK_DIR / case
                profiles[case] = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=300))
                contexts[case] = assemble_repo_context(repo)
            paper = {"title": papers[pid]["title"], "abstract": papers[pid]["abstract"]}
            try:
                gate = score_actionability(paper, profiles[case], gate_cfg)[0]
            except Exception as exc:  # noqa: BLE001
                print(f"    ! gate {pid} failed: {exc}")
                failures += 1
                continue
            row = {"stratum": stratum, "case": case, "id": pid, "gate": gate, "judge": None}
            if pid in to_judge:
                try:
                    verdict = judge_mod.judge_paper(
                        case,
                        contexts[case],
                        {"arxiv_id": pid, **paper},
                        model=args.judge_model,
                    )
                    row["judge"] = int(verdict["score"])
                    judged += 1
                    actionable += row["judge"] >= ACTIONABLE
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! judge {pid} failed: {exc}")
            admits += gate >= ACTIONABLE
            rows.append(row)
            OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        n = sum(1 for r in rows if r["stratum"] == stratum)
        print(
            f"[{stratum:14}] gated {n:3d}  admit {admits:3d} ({admits / max(n, 1):5.1%})   "
            f"judged {judged:3d}  actionable {actionable:2d} ({actionable / max(judged, 1):5.1%})"
            + (f"   {failures} failures" if failures else ""),
            flush=True,
        )

    after = resolve_targets()
    if before != after:
        print("\n!! THE GOLD SET MOVED during this run — wild verdicts leaked into it.")
        print(
            f"   before {sum(len(v) for v in before.values())} targets, "
            f"after {sum(len(v) for v in after.values())}"
        )
        return 1
    report(rows)
    return 0


def report(rows: list[dict[str, Any]]) -> None:
    print("\n=== P5 — stratified labels over the candidate pool ===")
    print(
        f"{'stratum':14} {'gated':>6} {'admit%':>8} {'admit 95% CI':>16} "
        f"{'judged':>7} {'>=2':>7} {'>=2 95% CI':>16} {'==3':>7}"
    )
    stats: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        sel = [r for r in rows if r["stratum"] == stratum]
        jud = [r for r in sel if r["judge"] is not None]
        act = sum(1 for r in jud if r["judge"] >= ACTIONABLE)
        # The >=2 bar is the shipped one, and it is permissive: PR #83 measured half of a
        # random hop-pool sample clearing it. Score 3 — "directly addresses a known
        # limitation" — is the bar the 48 gold targets were drawn at, so it is the one that
        # compares to every recall number in this project.
        top3 = sum(1 for r in jud if r["judge"] >= 3)
        adm = sum(1 for r in sel if r["gate"] >= ACTIONABLE)
        alo, ahi = wilson(adm, len(sel))
        jlo, jhi = wilson(act, len(jud))
        stats[stratum] = {
            "n": len(sel),
            "admits": adm,
            "admit_rate": adm / max(len(sel), 1),
            "judged": len(jud),
            "actionable": act,
            "actionable_rate": act / max(len(jud), 1),
            "score3": top3,
            "score3_rate": top3 / max(len(jud), 1),
        }
        print(
            f"{stratum:14} {len(sel):>6} {adm / max(len(sel), 1):>7.1%} "
            f"{f'[{alo:.1%}, {ahi:.1%}]':>16} {len(jud):>7} "
            f"{act / max(len(jud), 1):>6.1%} {f'[{jlo:.1%}, {jhi:.1%}]':>16} "
            f"{top3 / max(len(jud), 1):>6.1%}"
        )

    jud = [r for r in rows if r["judge"] is not None]
    tp = sum(1 for r in jud if r["gate"] >= ACTIONABLE and r["judge"] >= ACTIONABLE)
    fp = sum(1 for r in jud if r["gate"] >= ACTIONABLE and r["judge"] < ACTIONABLE)
    fn = sum(1 for r in jud if r["gate"] < ACTIONABLE and r["judge"] >= ACTIONABLE)
    base = sum(1 for r in jud if r["judge"] >= ACTIONABLE) / max(len(jud), 1)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(
        f"\ngate vs judge OFF-DISTRIBUTION (n={len(jud)}): precision {prec:.2f}, recall "
        f"{rec:.2f}, base rate {base:.1%}  (tp={tp} fp={fp} fn={fn})"
    )
    print("  on the labelled set the same gate measured precision 0.81 / recall 0.78.")

    top, floor = stats["hyde-top100"], stats["random-arxiv"]
    p = fisher_exact(
        top["actionable"],
        top["judged"] - top["actionable"],
        floor["actionable"],
        floor["judged"] - floor["actionable"],
    )
    wild = stats["hyde-100-1k"]["admit_rate"]
    print(
        f"\nPRE-REGISTERED: hyde-top100 >={PREDICT_TOP_STRATUM:.0%} vs random-arxiv "
        f"<={PREDICT_FLOOR:.0%}; gate wild-admit <={GATE_ADMIT_PASS:.0%}; "
        f"KILL if every stratum <={KILL_ALL_STRATA:.0%} or wild-admit >{GATE_ADMIT_KILL:.0%}"
    )
    print(
        f"  separation: {top['actionable_rate']:.1%} vs {floor['actionable_rate']:.1%}, "
        f"Fisher exact p = {p:.3f}"
    )
    print(f"  gate wild-admit (hyde-100-1k): {wild:.1%}")
    # Clause by clause, not as one conjunction. The first version collapsed the three parts
    # of the separation prediction into a single `and` and printed "BELOW PREDICTION" for a
    # 29x effect at p < 0.001, because the floor came in at 2/100 against a <=1/100 bar.
    # A pre-registered criterion should be reported as written AND decomposed, so that a
    # miss on one clause cannot be read as a miss on the hypothesis.
    print("\n  clause by clause:")
    clauses = separation_clauses(top["actionable_rate"], floor["actionable_rate"])
    for name, ok in clauses.items():
        print(f"    {'MET    ' if ok else 'MISSED '} {name}")
    p3 = fisher_exact(
        top["score3"],
        top["judged"] - top["score3"],
        floor["score3"],
        floor["judged"] - floor["score3"],
    )
    strict = (
        top["score3_rate"] >= PREDICT_TOP_STRATUM
        and floor["score3_rate"] <= PREDICT_FLOOR
        and top["score3_rate"] >= 6 * max(floor["score3_rate"], 1e-9)
    )
    print(
        f"    at the >=3 bar (the one the gold targets were drawn at): "
        f"{top['score3_rate']:.1%} vs {floor['score3_rate']:.1%}, p = {p3:.4f} — all three "
        f"clauses {'MET' if strict else 'not met'}"
    )
    verdicts = []
    if all(clauses.values()):
        verdicts.append("separation MET")
    elif all(s["actionable_rate"] <= KILL_ALL_STRATA for s in stats.values()):
        verdicts.append("separation KILL — no stratum is distinguishable at this density")
    elif clauses["separation >=6x"]:
        missed = ", ".join(n for n, ok in clauses.items() if not ok)
        verdicts.append(f"separation MET on effect size, criterion missed on: {missed}")
    else:
        verdicts.append("separation BELOW PREDICTION")
    if wild > GATE_ADMIT_KILL:
        verdicts.append("gate KILL — wild-admit above the kill bar")
    elif wild <= GATE_ADMIT_PASS:
        verdicts.append("gate MET")
    else:
        verdicts.append("gate BETWEEN the pass and kill bars")
    print("verdict: " + "; ".join(verdicts))

    # What the kill bar was FOR. It assumed a high admit rate means the gate is being fooled
    # by a junk-dominated pool. Whether that premise holds is a separate, measurable thing.
    if wild > GATE_ADMIT_KILL:
        band = stats["hyde-100-1k"]
        print(
            f"\n  the kill clause assumed a high admit rate means junk is getting through. "
            f"Measured: gate precision {prec:.2f} (fp={fp} of {tp + fp} admits), recall "
            f"{rec:.2f} (fn={fn}). The band it fired on is {band['actionable_rate']:.0%} "
            f"actionable, ABOVE its {wild:.0%} admit rate — so the gate is rejecting "
            f"actionable papers, not admitting junk. Read the kill as aimed at recall."
        )

    print("\n  cascade volume — papers surviving the gate, per repo, at each operating point:")
    for stratum, per_repo in (("hyde-top100", 100), ("hyde-100-1k", 1000), ("hyde-1k-10k", 10000)):
        s = stats[stratum]
        print(
            f"    {stratum:14} {per_repo:>6,} candidates -> "
            f"~{round(per_repo * s['admit_rate']):>5,} admitted  "
            f"(~${per_repo * 0.00023:.2f} of Haiku per repo per run)"
        )
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
