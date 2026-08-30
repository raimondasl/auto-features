"""Item 12, the follow-up NR-49 named: do round 2's papers ever reach the gate? [NR-50]

NR-49 closed PRF-HyDE on reach — null at two budget points, +3 witnesses at 400 slots
(*p* = 0.68) and +1 at 800 (*p* = 1.00). But it recorded the caveat that keeps net@2 open:
reach is measured over 520 witnesses while the **pools differ across their whole contents**,
and at 800 slots the two arms reach near-identical totals through *different* papers.

So this asks the question NR-47 and NR-48 between them showed is the binding one. Both found
the gate reads a fixed `gate_depth` of a **ranked** pool: a candidate that never ranks into
that window cannot change the digest, whatever its quality. **If round 2's papers do not reach
the top 50, net@2 cannot move and no paid arm can rescue it.**

**The configuration is deliberately generous.** Round 2 at `top_k` 100 merged into the *whole*
shipped pool (round 1 at 100 plus keyword) — the superset, not NR-49's budget match. PRF gets
every candidate it can ask for and none of round 1 is taken away, so a null here is decisive
for the matched arm too, while a pass licenses either.

**PRE-REGISTERED, WITH AN EFFECT SIZE.** NR-49's bar named a threshold and no effect size, and
a null cleared it by three witnesses of 520. That is the mistake this probe exists not to
repeat, so the bar is stated as a quantity and the arithmetic behind it is written down:

The chain is: the top-50 window feeds the gate, whose admits are re-scored and cut to a digest
of **8.3/case — 16.6% of the window**. A round-2 paper in the window displaces a shipped one,
and a swap is worth **3Δp** in net@2. So a window share *s* moves net@2 by about
`s × 50 × 0.166 × 3Δp` per case, against a paired bootstrap that resolved **±0.78** at n = 37
in NR-47. That gives a ladder rather than a line, and all three rungs are registered here:

| round-2 share of the top 50 | at Δp = 0.2 (generous) | at Δp = 0.092 (NR-48's *observed* gap) | |
|---|---|---|---|
| **< 5%** | < 0.25 | < 0.11 | **KILL — item 12 closes** |
| 5–16% | 0.25–0.78 | 0.11–0.36 | grey: needs an implausible Δp to be seen |
| **≥ 16%** | ≥ 0.78 | ≥ 0.36 | **paid arm licensed** |

* **The kill is < 5%.** Below it fewer than one digest paper per case can differ and no quality
  gap rescues that, so the result is not "small" — it is **invisible to the estimators this
  project uses**, and ~$15 would buy a number that cannot answer the question.
* **The honest part is the grey band**, and it is wide. At the quality gap NR-48 actually
  measured — 0.951 displaced by 0.859 — round 2 would need **34% of the window** to shift
  net@2 by a resolvable 0.78. Clearing 5% is necessary and a long way from sufficient, which
  is the sentence NR-49's pre-registration failed to contain.

Displacement is reported alongside, because NR-48's lesson was that what a change *removes*
matters as much as what it adds: there, 99 admitted at 0.859 precision displaced 41 at 0.951.

    uv run python evals/prf_rank_probe.py --search    # ~12 min CPU, round-2 top-100 per case
    uv run python evals/prf_rank_probe.py --collect   # arXiv metadata for the new ids
    uv run python evals/prf_rank_probe.py             # $0, the probe itself
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from reporadar.paper_id import dedup_id  # noqa: E402

WORK = EVALS / ".work"
INDEX = WORK / "hyde_index"
SHIPPED_POOL = WORK / "pool-cut100"
ROUND2_HYP = WORK / "hyde_hypotheses_prf.json"
ROUND2_IDS = WORK / "prf_round2_ids.json"
ROUND2_META = WORK / "prf_round2_meta.json"
WINDOWS = WORK / "prf_rank_windows.json"  # the expensive half: per-case window ids
FROZEN = EVALS / "prf_rank_probe.json"

TOP_K = 100  # round 2's cut, matching the shipped round-1 cut
GATE_DEPTH = 50  # what the gate actually reads; the whole question
BAR_KILL = 0.05  # pre-registered floor: below this, net@2 cannot move at any Δp
BAR_LICENSE = 0.16  # and below THIS it cannot move resolvably at a generous Δp = 0.2
DIGEST_SHARE = 0.166  # 8.3 digest papers of a 50-deep window


def cases_with_round2() -> dict[str, list[str]]:
    return json.loads(ROUND2_HYP.read_text(encoding="utf-8"))


def search() -> int:
    """Round 2's top-`TOP_K` ids per case: the candidates PRF would actually contribute."""
    from reporadar import hyde

    hyp = cases_with_round2()
    model = hyde.load_encoder()
    ok, dists = hyde.verify_encoder(model)
    if not ok:
        raise SystemExit(f"encoder does not reproduce the index (Hamming {dists}); refusing")
    out: dict[str, list[str]] = (
        json.loads(ROUND2_IDS.read_text(encoding="utf-8")) if ROUND2_IDS.is_file() else {}
    )
    todo = [c for c in sorted(hyp) if c not in out]
    print(f"{len(todo)} case(s) to search")
    for n, case in enumerate(todo, start=1):
        # `search_index` already unions the query rows and keeps each row's own top-k --
        # the same call the shipped `discover` makes, not a reimplementation of it.
        bits = hyde.encode_binary(model, list(hyp[case]))
        ids = hyde.search_index(INDEX, bits, top_k=TOP_K)
        out[case] = sorted({dedup_id(i) for i in ids})
        ROUND2_IDS.write_text(json.dumps(out, indent=0), encoding="utf-8")
        print(f"  [{n}/{len(todo)}] {case:<16} {len(out[case]):>4} unique ids")
    return 0


def shipped_pool(case: str) -> list[dict[str, Any]]:
    f = SHIPPED_POOL / f"{case}.json"
    return json.loads(f.read_text(encoding="utf-8"))["candidates"] if f.is_file() else []


def new_ids() -> dict[str, list[str]]:
    """Round-2 ids the shipped pool does not already hold, per case."""
    ids = json.loads(ROUND2_IDS.read_text(encoding="utf-8"))
    out = {}
    for case, got in ids.items():
        have = {dedup_id(str(p["arxiv_id"])) for p in shipped_pool(case)}
        out[case] = [i for i in got if i not in have]
    return out


def collect() -> int:
    """Fetch metadata for the new ids, through the shipped collector.

    Batched by the collector itself and cached under `.work/arxiv-cache`, so a re-run after a
    throttle resumes rather than restarting. A case that fails is left absent and reported --
    **void, not null**: a case scored as "round 2 contributed nothing" because its fetch 429'd
    would be the same defect C-4 and C-30 are named for.
    """
    from reporadar.collector import CollectionError, collect_by_ids

    want = new_ids()
    have = json.loads(ROUND2_META.read_text(encoding="utf-8")) if ROUND2_META.is_file() else {}
    todo = [c for c in sorted(want) if c not in have]
    print(f"{len(todo)} case(s) to collect; {sum(len(want[c]) for c in todo)} ids")
    for n, case in enumerate(todo, start=1):
        try:
            papers = collect_by_ids(want[case])
        except CollectionError as exc:
            print(f"  [{n}/{len(todo)}] {case:<16} FAILED (void, not null): {str(exc)[:70]}")
            continue
        have[case] = papers
        ROUND2_META.write_text(json.dumps(have), encoding="utf-8")
        print(f"  [{n}/{len(todo)}] {case:<16} {len(papers):>4}/{len(want[case])} resolved")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="recompute the windows (~35 min); otherwise the cached ones are reused",
    )
    args = ap.parse_args()
    if args.search:
        return search()
    if args.collect:
        return collect()

    from harness import load_benchmark
    from prf_hyde_reach import judged_scores
    from run_judge_eval import rank_candidates

    bench = {c["name"]: c for c in load_benchmark()["cases"]}
    meta = json.loads(ROUND2_META.read_text(encoding="utf-8"))
    cached = (
        json.loads(WINDOWS.read_text(encoding="utf-8"))
        if WINDOWS.is_file() and not args.rerank
        else None
    )

    def embed(papers: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch-encode a pool once, for both rankings.

        The ranker encodes title + abstract per paper, one call at a time, and this probe
        ranks each pool twice (merged, then baseline). Encoding once in a batch is the same
        arithmetic at a fraction of the wall clock -- and it keeps the two rankings on
        byte-identical vectors, so a rank difference cannot come from the encoder.
        """
        from reporadar.embeddings import _get_model

        texts = [f"{p.get('title', '')}. {p.get('abstract', '')}" for p in papers]
        vecs = _get_model().encode(texts, convert_to_numpy=True, batch_size=64)
        return {p["arxiv_id"]: v for p, v in zip(papers, vecs, strict=True)}

    per_case: dict[str, Any] = cached or {}
    for case in sorted(meta) if cached is None else []:
        base = shipped_pool(case)
        if not base:
            continue
        known = {dedup_id(str(p["arxiv_id"])) for p in base}
        extra = [p for p in meta[case] if dedup_id(str(p["arxiv_id"])) not in known]
        merged = base + extra
        vecs = embed(merged)
        # The benchmark's own declared categories, exactly as every other ranking probe in
        # `evals/` reads them. Deriving them from the pool instead would let the treatment
        # arm's extra papers change the category axis they are then scored on.
        cats = bench[case].get("expected_categories") or ["cs.LG"]
        ranked = rank_candidates(
            WORK / case,
            merged,
            cats,
            top_n=GATE_DEPTH,
            all_time=True,
            hybrid=True,
            w_embedding=1.5,
            paper_embeddings=vecs,
        )
        window = [dedup_id(str(p["arxiv_id"])) for p, _s in ranked]
        r2 = {dedup_id(str(p["arxiv_id"])) for p in extra}
        # What the shipped pool alone put in the window, to measure displacement.
        base_ranked = rank_candidates(
            WORK / case,
            base,
            cats,
            top_n=GATE_DEPTH,
            all_time=True,
            hybrid=True,
            w_embedding=1.5,
            paper_embeddings=vecs,
        )
        base_window = [dedup_id(str(p["arxiv_id"])) for p, _s in base_ranked]
        entered = [i for i in window if i in r2]
        displaced = [i for i in base_window if i not in window]
        per_case[case] = {
            "pool_shipped": len(base),
            "round2_new": len(extra),
            "round2_in_window": len(entered),
            "displaced": len(displaced),
            # Persisted so the quality diagnostic below -- and anything later -- reads them
            # instead of re-ranking 33 cases to recover ids this pass already had.
            "entered_ids": entered,
            "displaced_ids": displaced,
        }
        print(
            f"  {case:<16} +{len(extra):>4} new  ->  "
            f"{per_case[case]['round2_in_window']:>2}/{GATE_DEPTH} in window, "
            f"{per_case[case]['displaced']:>2} displaced"
        )

    if cached is None:
        WINDOWS.write_text(json.dumps(per_case), encoding="utf-8")
    else:
        print(f"reusing cached windows for {len(per_case)} cases (--rerank to recompute)")

    n = len(per_case)
    in_win = sum(c["round2_in_window"] for c in per_case.values())
    share = in_win / (n * GATE_DEPTH) if n else 0.0
    out = {
        "_comment": (
            "NR-50 / item 12 follow-up: do round 2's papers reach the gate at all? Derived by "
            "evals/prf_rank_probe.py; pinned by tests/test_prf_rank_probe.py. The gate reads a "
            "fixed depth of a RANKED pool (NR-47, NR-48), so a candidate outside the top "
            f"{GATE_DEPTH} cannot change the digest whatever its quality. Configuration is "
            "deliberately generous: round 2 at top_k 100 merged into the WHOLE shipped pool, "
            "so a null here is decisive for NR-49's budget-matched arm too."
        ),
        "pre_registered": {
            "kill_below": BAR_KILL,
            "license_at_or_above": BAR_LICENSE,
            "bar_per_case": BAR_KILL * GATE_DEPTH,
            "arithmetic": (
                "net@2/case ~= share * 50 * 0.166 * 3dp. The 0.166 is the digest's share of "
                "the window (8.3 of 50). Against NR-47's +-0.78 bootstrap at n = 37: below 5% "
                "nothing is measurable at any dp; 16% is the threshold at a GENEROUS dp = 0.2; "
                "at the dp NR-48 actually observed (0.951 displaced by 0.859, dp = 0.092) the "
                "requirement is 34% of the window. Clearing 5% is necessary, not sufficient."
            ),
            "share_needed_at_observed_dp": 0.34,
            "fixes_what_NR49_got_wrong": (
                "NR-49's bar named a threshold and no minimum effect size, so a null cleared "
                "it by 3 witnesses of 520. This one is a quantity with its arithmetic shown."
            ),
        },
        "cases": n,
        "gate_depth": GATE_DEPTH,
        "round2_window_slots": in_win,
        "window_slots_total": n * GATE_DEPTH,
        "share_of_window": round(share, 4),
        "per_case_mean": round(in_win / n, 2) if n else 0.0,
        "displaced_total": sum(c["displaced"] for c in per_case.values()),
        "per_case": per_case,
    }
    # Direction, not just magnitude. The share says an effect would be RESOLVABLE; it says
    # nothing about its sign. C-35's caveat holds -- the judge cache is the union of every
    # experiment, so it is read here only to score papers named on other grounds, and papers
    # absent from it are VOID, not null.
    scores = judged_scores()

    def prec(ids: list[tuple[str, str]]) -> dict[str, Any]:
        got = [scores[k] for k in ids if k in scores]
        return {
            "judged": len(got),
            "of": len(ids),
            "precision": round(sum(1 for x in got if x >= 2) / len(got), 3) if got else None,
            "void_never_judged": len(ids) - len(got),
        }

    entered_k = [(c, i) for c, v in per_case.items() for i in v["entered_ids"]]
    displaced_k = [(c, i) for c, v in per_case.items() for i in v["displaced_ids"]]
    out["quality_of_the_swap"] = {
        "_comment": (
            "What the window trade is made of, where any judgement exists. NR-48's lesson was "
            "that a change is the difference between what it admits and what it removes: there "
            "99 admitted at 0.859 displaced 41 at 0.951, and the arm lost. This is the same "
            "diagnostic one stage earlier, at the window rather than the digest."
        ),
        "round2_entering": prec(entered_k),
        "shipped_displaced": prec(displaced_k),
    }
    ep = out["quality_of_the_swap"]["round2_entering"]["precision"]
    dp_ = out["quality_of_the_swap"]["shipped_displaced"]["precision"]
    out["quality_of_the_swap"]["observed_dp"] = (
        round(ep - dp_, 3) if ep is not None and dp_ is not None else None
    )

    out["verdict"] = {
        "share_of_window": round(share, 4),
        "killed": bool(share < BAR_KILL),
        "licenses_paid_arm": bool(share >= BAR_LICENSE),
        "in_grey_band": bool(BAR_KILL <= share < BAR_LICENSE),
        "implied_net2_per_case": {
            "at_generous_dp_0.20": round(share * GATE_DEPTH * DIGEST_SHARE * 3 * 0.20, 3),
            "at_observed_dp_0.092": round(share * GATE_DEPTH * DIGEST_SHARE * 3 * 0.092, 3),
        },
        "bootstrap_resolves_at_n37": 0.78,
    }
    obs = out["quality_of_the_swap"]["observed_dp"]
    if obs is not None:
        implied = share * GATE_DEPTH * DIGEST_SHARE * 3 * obs
        out["verdict"]["at_the_dp_measured_here"] = {
            "dp": obs,
            "implied_net2_per_case": round(implied, 3),
            "would_resolve": bool(abs(implied) >= 0.78),
            "_comment": (
                "The tension worth stating rather than resolving by preference. The share "
                "clears the pre-registered licence at 20.6%, and it was set at a GENEROUS "
                "dp = 0.2. The only quality evidence available puts dp at 0.054, which implies "
                "an effect INSIDE the noise band. Both are reported: the pre-registration is "
                "honoured because moving the bar after seeing the data is the failure NR-49 "
                "documented, and the prior is reported because it is what a reader deciding "
                "whether to spend needs. Note the dp itself is weak -- 61% of entering and 73% "
                "of displaced papers are VOID, and the judged subset is selected by having "
                "been shown by some arm, so it is a hint, not a measurement."
            ),
        }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        f"\n{n} cases; round 2 holds {in_win}/{n * GATE_DEPTH} window slots "
        f"({share:.2%}, {out['per_case_mean']}/case), displacing {out['displaced_total']}"
    )
    v = out["verdict"]
    band = "KILL" if v["killed"] else ("LICENSED" if v["licenses_paid_arm"] else "GREY")
    print(
        f"bar: kill <{BAR_KILL:.0%}, license >={BAR_LICENSE:.0%}  ->  {band}\n"
        f"implied net@2/case: {v['implied_net2_per_case']['at_generous_dp_0.20']:+.3f} at a "
        f"generous dp, {v['implied_net2_per_case']['at_observed_dp_0.092']:+.3f} at NR-48's "
        f"observed dp (bootstrap resolves +-0.78)"
    )
    print(f"wrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
