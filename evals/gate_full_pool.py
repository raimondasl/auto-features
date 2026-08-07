"""What happens if the LLM gate sees the WHOLE candidate pool instead of the ranker's top 15?

`triage.top_k` is 15. Nobody wrote down why — it was set in the Feature 6 commit with no
rationale — and the obvious objection to raising it, cost, does not survive contact with the
arithmetic: the gate is ~$0.00023 a paper.

**Corrected while building the pools (2026-08-07):** the per-repo pool is **~250 papers, not
~2,000**. The 2,030 figure quoted everywhere came from `diagnose_pool`'s "papers fetched
across 12 cases: 2030" — a cross-case TOTAL, ~170 each. Measured here: 56 to 296 per repo,
with three cases landing on 296, which is the ceiling of `#queries x max_results_per_query`
minus duplicates. So gating an entire pool is **$0.06 per repo per run** and ~7 minutes
sequential, and `top_k: 15` currently shows the gate **6% of the pool**.

The reason to care is that the filter it replaces is measured **at chance**.
`diagnose_ranker.py` found ranks 1-10 and 11-50 indistinguishable (31% vs 33% actionable), so
the 15 papers the gate currently sees are close to a random draw from the ranker's top 50, and
an actionable paper sitting at rank 800 can never reach the digest.

**What is unmeasured is the gate itself on this distribution.** Every precision figure this
project has — 0.92 on the labelled set, 0.97 in P5 — comes from papers surfaced by the ranker,
by Opus, or by HyDE. All strong strata. The one wild measurement, P5's `random-arxiv`, is the
opposite extreme (0 of 200 admitted). The real pool is neither: a few hundred
topically-adjacent papers from repo-derived queries, and nothing has ever labelled it.

Three numbers decide whether raising `top_k` is an improvement or a flood:

  1. **admit rate on the full pool** — at 13% you get ~260 admits per repo, at 40% the gate is
     not filtering at all and `top_k` is load-bearing for a reason nobody wrote down
  2. **gate precision off this distribution** — is 0.92-0.97 a property of the gate or of the
     strata it was measured on
  3. **wall-clock per call** — the real constraint. ~250 sequential calls at ~1.6 s is about
     7 minutes per repo; the fix is concurrency, and its size depends on this number

    uv run python evals/gate_full_pool.py --build     # fetch real pools, free, cached
    uv run python evals/gate_full_pool.py --dry-run   # sample + cost, $0
    uv run python evals/gate_full_pool.py             # ~$0.07 gate + ~$3.2 judge

**Pre-registered before running (2026-08-07).**

  * PREDICTION: admit rate on the full pool is **10-25%**, bracketing the 21% measured on the
    labelled set and the 13% of P5's thin strata; gate precision stays **>=0.80** — high
    enough that admitting 300 papers beats picking 15 at chance.
  * KILL for the whole "raise top_k" direction: admit rate **>40%** (the gate is not a filter
    at pool scale, and a digest fed by it would be flooded) OR precision **<0.60** (the 0.92+
    figures were a property of the strata, not the gate).

Judge verdicts are written to the shared cache under the SHIPPED rubric, which is correct —
they are new labels for papers nobody has labelled. `resolve_targets()` is asserted unchanged
across the run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import judge as judge_mod  # noqa: E402
import yaml  # noqa: E402
from diagnose_triage import _load_env  # noqa: E402
from harness import (  # noqa: E402
    WORK_DIR,
    assemble_repo_context,
    collect_live_papers,
    profile_case_repo,
)
from label_pool import balanced_draw, wilson  # noqa: E402

from reporadar.config import ProfilerConfig, SuggestionsConfig  # noqa: E402
from reporadar.profiler import profile_repo  # noqa: E402
from reporadar.triage import score_actionability  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
POOLS = WORK / "full_pool"
OUT = WORK / "gate_full_pool.json"
BENCH = EVALS / "benchmark.yaml"

ACTIONABLE = 2
SEED = 20260807
GATE_N = 300
JUDGE_NON_ADMITS = 40
JUDGE_ADMIT_CAP = 60
# Pre-registered, see module docstring.
PREDICT_ADMIT_LO, PREDICT_ADMIT_HI = 0.10, 0.25
PREDICT_PRECISION = 0.80
KILL_ADMIT = 0.40
KILL_PRECISION = 0.60


def build_pools(refresh: bool = False, pause: float = 10.0) -> None:
    """Fetch each case's real candidate pool and persist it.

    This is the pool `rr update` builds today: the same query construction, the same
    all-time relevance-first discovery that is now the shipped default. Cached because it is
    a few thousand live arXiv requests and the question does not change between runs.
    """
    POOLS.mkdir(parents=True, exist_ok=True)
    bench = yaml.safe_load(BENCH.read_text(encoding="utf-8"))
    for case in bench["cases"]:
        name = case["name"]
        out = POOLS / f"{name}.jsonl"
        if out.is_file() and not refresh:
            continue
        repo = WORK_DIR / name
        if not repo.is_dir():
            print(f"[{name:10}] no clone — skipping")
            continue
        t0 = time.perf_counter()
        profile = profile_case_repo(repo)
        try:
            papers = collect_live_papers(
                profile,
                case.get("expected_categories") or ["cs.LG"],
                sources=["arxiv"],
                keys={},
                lookback_days=36500,
                sort_by="relevance",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{name:10}] fetch failed: {exc}")
            continue
        # The collector swallows a rate-limited query and returns what it got, so a 429
        # storm yields an EMPTY pool with no exception. Persisting that would cache a lie:
        # the next run skips the case as "already built" and the case silently drops out of
        # every downstream number. Same failure class as the hop builder's failed_chunks.
        if not papers:
            print(f"[{name:10}] fetch returned NOTHING (rate limited?) — refusing to cache")
            continue
        seen: set[str] = set()
        rows = []
        for p in papers:
            pid = str(p.get("arxiv_id", "")).split("v")[0]
            if not pid or pid in seen or not p.get("abstract"):
                continue
            seen.add(pid)
            rows.append({"id": pid, "title": p.get("title", ""), "abstract": p["abstract"]})
        out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        print(
            f"[{name:10}] pool={len(rows):5,} papers  {time.perf_counter() - t0:5.1f}s",
            flush=True,
        )
        # arXiv throttled this script at ~15 consecutive cases (~90 queries back to back)
        # and returned 429 for every query after that. The per-query retry inside the
        # collector cannot help once the whole client is throttled, so the fix is to be
        # slower between cases rather than more persistent within one.
        time.sleep(pause)


def load_pools() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(POOLS.glob("*.jsonl")):
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        if rows:
            out[path.stem] = rows
    return out


def report(rows: list[dict[str, Any]], pools: dict[str, int]) -> None:
    n = len(rows)
    admits = [r for r in rows if r["gate"] >= ACTIONABLE]
    rate = len(admits) / max(n, 1)
    lo, hi = wilson(len(admits), n)
    print(f"\n=== gate on the FULL candidate pool — {n} papers, {len(pools)} repos ===")
    print(f"admit rate  {rate:.1%}  95% CI [{lo:.1%}, {hi:.1%}]   ({len(admits)}/{n})")

    print(f"\n{'case':12} {'pool':>7} {'sampled':>8} {'admits':>7} {'rate':>7}")
    for case in sorted(pools):
        sel = [r for r in rows if r["case"] == case]
        if not sel:
            continue
        a = sum(1 for r in sel if r["gate"] >= ACTIONABLE)
        print(f"{case:12} {pools[case]:>7,} {len(sel):>8} {a:>7} {a / len(sel):>6.0%}")

    judged = [r for r in rows if r.get("judge") is not None]
    ja = [r for r in judged if r["gate"] >= ACTIONABLE]
    jn = [r for r in judged if r["gate"] < ACTIONABLE]
    tp = sum(1 for r in ja if r["judge"] >= ACTIONABLE)
    fp = len(ja) - tp
    fn = sum(1 for r in jn if r["judge"] >= ACTIONABLE)
    prec = tp / max(len(ja), 1)
    plo, phi = wilson(tp, len(ja))
    print(
        f"\ngate vs judge on this distribution (n={len(judged)}):"
        f"\n  precision {prec:.2f}  95% CI [{plo:.2f}, {phi:.2f}]   (tp={tp} fp={fp})"
        f"\n  of {len(jn)} judged NON-admits, {fn} were actionable "
        f"({fn / max(len(jn), 1):.0%} — the recall leak)"
    )
    print("  for comparison: 0.92 on the labelled set, 0.97 on P5's stratified wild sample")

    times = [r["seconds"] for r in rows if r.get("seconds")]
    if times:
        times.sort()
        med = times[len(times) // 2]
        mean_pool = sum(pools.values()) / max(len(pools), 1)
        print(
            f"\nlatency: median {med:.2f}s per call, mean {sum(times) / len(times):.2f}s"
            f"\n  a {mean_pool:,.0f}-paper pool costs ~${mean_pool * 0.00023:.2f} and "
            f"~{mean_pool * med / 60:.0f} min sequential, ~{mean_pool * med / 60 / 16:.0f} min "
            f"at 16-way concurrency"
        )

    print(
        f"\nPRE-REGISTERED: admit {PREDICT_ADMIT_LO:.0%}-{PREDICT_ADMIT_HI:.0%} and precision "
        f">={PREDICT_PRECISION:.2f}; KILL if admit >{KILL_ADMIT:.0%} or precision "
        f"<{KILL_PRECISION:.2f}"
    )
    if rate > KILL_ADMIT:
        verdict = f"KILL — {rate:.0%} admit means the gate is not a filter at pool scale"
    elif len(ja) and prec < KILL_PRECISION:
        verdict = f"KILL — precision {prec:.2f} was a property of the strata, not the gate"
    elif PREDICT_ADMIT_LO <= rate <= PREDICT_ADMIT_HI and prec >= PREDICT_PRECISION:
        verdict = "MET — raising top_k is an improvement, and the volume is manageable"
    else:
        parts = []
        if not (PREDICT_ADMIT_LO <= rate <= PREDICT_ADMIT_HI):
            parts.append(f"admit {rate:.0%} outside the predicted band")
        if prec < PREDICT_PRECISION:
            parts.append(f"precision {prec:.2f} below {PREDICT_PRECISION:.2f}")
        verdict = "BELOW PREDICTION: " + "; ".join(parts)
    print(f"verdict: {verdict}")
    print(f"\nwritten to {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="fetch and cache the real pools")
    ap.add_argument("--refresh", action="store_true", help="re-fetch pools even if cached")
    ap.add_argument("--pause", type=float, default=10.0, help="seconds between cases (arXiv 429s)")
    ap.add_argument("--n", type=int, default=GATE_N)
    ap.add_argument("--dry-run", action="store_true", help="sample + cost, call nothing")
    ap.add_argument("--report", action="store_true", help="re-derive from saved rows, $0")
    ap.add_argument("--gate-model", default="claude-haiku-4-5")
    ap.add_argument("--judge-model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()
    _load_env()

    if args.build:
        build_pools(args.refresh, args.pause)
        return 0

    pools = load_pools()
    if not pools:
        print("no pools — run with --build first")
        return 1
    sizes = {c: len(v) for c, v in pools.items()}
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8")), sizes)
        return 0

    by_id = {(c, r["id"]): r for c, rows in pools.items() for r in rows}
    rng = random.Random(SEED)
    picked = balanced_draw([(c, r["id"]) for c, rows in pools.items() for r in rows], args.n, rng)
    counts = Counter(c for c, _ in picked)
    print(f"pools: {sum(sizes.values()):,} papers across {len(sizes)} repos")
    print(f"  sizes: {dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:5])} ...")
    print(f"sampled {len(picked)} across {len(counts)} repos (~{len(picked) * 0.00023:.2f} gate)")
    if args.dry_run:
        return 0

    from build_hop_pool import resolve_targets

    before = resolve_targets()
    gate_cfg = SuggestionsConfig(provider="claude", claude_model=args.gate_model, timeout=60)
    profiles: dict[str, Any] = {}
    contexts: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for case, pid in picked:
        if case not in profiles:
            profiles[case] = profile_repo(
                WORK_DIR / case, profiler_cfg=ProfilerConfig(prose_chars=300)
            )
            contexts[case] = assemble_repo_context(WORK_DIR / case)
        paper = by_id[(case, pid)]
        t0 = time.perf_counter()
        try:
            gate = score_actionability(paper, profiles[case], gate_cfg)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"    ! gate {case}/{pid} failed: {exc}")
            continue
        rows.append(
            {
                "case": case,
                "id": pid,
                "gate": gate,
                "judge": None,
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )
        if len(rows) % 50 == 0:
            print(f"  gated {len(rows)}/{len(picked)}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    admits = [r for r in rows if r["gate"] >= ACTIONABLE]
    non = [r for r in rows if r["gate"] < ACTIONABLE]
    rng.shuffle(non)
    to_judge = admits[:JUDGE_ADMIT_CAP] + non[:JUDGE_NON_ADMITS]
    print(
        f"\n{len(admits)} admitted of {len(rows)}; judging {len(to_judge)} "
        f"(~${len(to_judge) * 0.032:.2f})",
        flush=True,
    )
    for row in to_judge:
        paper = by_id[(row["case"], row["id"])]
        try:
            verdict = judge_mod.judge_paper(
                row["case"],
                contexts[row["case"]],
                {"arxiv_id": row["id"], **paper},
                model=args.judge_model,
            )
            row["judge"] = int(verdict["score"])
        except Exception as exc:  # noqa: BLE001
            print(f"    ! judge {row['id']} failed: {exc}")
        OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if resolve_targets() != before:
        print("\n!! THE GOLD SET MOVED — new verdicts leaked into the target list.")
        return 1
    report(rows, sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
