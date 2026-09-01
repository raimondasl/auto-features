"""How much of NR-52's cross-judge disagreement is Sonnet disagreeing with itself? [NR-53]

NR-52 reported a Sonnet-only margin of −3.41 against GPT's +0.32 and called the sign flip the
informative reading. Checking the code afterwards turned up a confound that undercuts it:
**`_call_claude` sends no temperature**, so the Anthropic default (1.0) applies and every
Sonnet verdict is a *sample*, while the GPT judge runs at `temperature=0` and is greedy.

So an unknown share of the observed disagreement is not a judge difference at all. This
measures that share the only way it can be measured — by asking the same judge the same
question twice, at the settings NR-52 actually used.

**This must run BEFORE the temperature is fixed.** At temperature 0 self-agreement is trivially
~1.0 and the quantity becomes unrecoverable; the 853 cached verdicts could never be
characterised again. It is also the cheap gate on the expensive repair: ~$3 here decides
whether re-judging 663 papers at temperature 0 (~$10) is necessary or pointless.

**The decision-relevant statistic is the LABEL FLIP RATE, not raw kappa.** NR-52's labels are
thresholds on the score — `son >= 1` for consensus, `son >= 2` for sonnet-only — so a 2→3
redraw changes nothing while a 1→2 redraw flips one label and not the other. Flip rate is what
propagates into a margin; kappa is reported alongside because it is what the literature quotes.

**PRE-REGISTERED, written before any replicate was drawn.**

* Population: the 663 papers that fed NR-52 (306 shipped + 357 Opus 5), sampled at random so
  the estimate is representative of the margin rather than of any score band.
* **PASS (noise is modest):** binary self-agreement kappa at the `>=2` cut **>= 0.6** AND
  sonnet-only label flip rate **<= 10%**. NR-52's flip is then a real judge difference, the
  temperature fix is hygiene, and the 663 verdicts need not be re-bought.
* **KILL (NR-52 is noise-dominated):** kappa **< 0.4** OR flip rate **>= 20%**. The Sonnet-only
  margin must then be re-derived at temperature 0 before it is cited anywhere, and NR-52's
  write-up needs a correction rather than a footnote.
* Between those: grey, reported with the implied margin perturbation and decided in the open.

**The perturbation arithmetic, fixed in advance.** A flipped label moves that paper's net@2
contribution by 3 (+1 becomes −2, or the reverse). With flip rate *f* over ~9 shown papers per
case, the per-case sd injected into a margin is about `3 * sqrt(9 * f * (1-f))` — **1.8 at
f = 0.01, 3.6 at f = 0.2**. The Sonnet-only margin is −3.41. That is the comparison the grey
band is decided on, and stating it now is the point: at a flip rate of even a few percent the
margin is not resolvable, and that would be true whichever way the sign fell.

Replicates are cached under a distinct model directory via `second_verdict(cache_as=...)`, so
the original draw is never overwritten and the two are independently addressable.

    uv run python evals/sonnet_self_agreement.py --plan     # $0: the sample, no calls
    uv run python evals/sonnet_self_agreement.py --judge    # ~$3, resumable
    uv run python evals/sonnet_self_agreement.py            # $0: agreement + flip rates
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

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from rung1_second_judge import (  # noqa: E402
    cached_sonnet,
    opus5_arm,
    pool_metadata,
    shipped_arm,
    son_of,
)
from second_judge import DEFAULT_MODEL, cohens_kappa, quadratic_kappa, verify_contexts  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

REPLICATE_TAG = f"{DEFAULT_MODEL}#replicate"
SEED = 20260831
SAMPLE_N = 200
FROZEN = EVALS / "sonnet_self_agreement.json"

PASS_KAPPA, PASS_FLIP = 0.60, 0.10
KILL_KAPPA, KILL_FLIP = 0.40, 0.20


def population() -> list[tuple[str, str, dict[str, Any], str]]:
    """(case, paper_id, paper, arm) for every paper that fed NR-52, in a stable order."""
    ship, opus = shipped_arm(), opus5_arm()
    cases = sorted(set(ship) & set(opus))
    rows = []
    for case in cases:
        for arm, picks in (("shipped", ship[case]), ("opus5", opus[case])):
            for p in picks:
                rows.append((case, str(p["arxiv_id"]), p, arm))
    return rows


def draw_sample(n: int = SAMPLE_N) -> list[tuple[str, str, dict[str, Any], str]]:
    """Random, not stratified.

    The question is how much noise contaminated the MARGIN, and the margin weights papers as
    they actually appear. A stratified sample would answer a different question (where noise
    lives by score band), which is reported afterwards as a description rather than sampled for.
    """
    rows = [r for r in population() if son_of(cached_sonnet(), r[0], r[1]) is not None]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


def plan() -> int:
    rows = draw_sample()
    by_arm = Counter(r[3] for r in rows)
    print(f"population with an original Sonnet verdict: {len(population())} papers")
    print(f"sample: {len(rows)}  {dict(by_arm)}")
    print(f"replicate cache: .work/second_judge/{REPLICATE_TAG}")
    print(f"estimated cost: ~${len(rows) * 0.012:.0f}-{len(rows) * 0.025:.0f}")
    return 0


def judge() -> int:
    from run_judge_eval import load_dotenv
    from second_judge import second_verdict
    from verify import resolve_references

    load_dotenv(EVALS / ".env")
    rows = draw_sample()
    contexts, drifted = verify_contexts(sorted({r[0] for r in rows}))
    if drifted:
        print(f"EXCLUDED (context drift): {sorted(drifted)}")
    done = failed = 0
    by_case: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
    for case, pid, paper, arm in rows:
        by_case.setdefault(case, []).append((pid, paper, arm))

    for n, (case, items) in enumerate(sorted(by_case.items()), start=1):
        if case not in contexts:
            continue
        ctx = contexts[case]
        meta = pool_metadata(case)
        need_resolve = [
            (pid, paper) for pid, paper, arm in items if not (meta.get(pid) or {}).get("abstract")
        ]
        resolved: dict[str, dict[str, Any]] = {}
        if need_resolve:
            try:
                got, *_ = resolve_references([pid for pid, _ in need_resolve], [])
                # dedup_id on both sides: the resolver returns VERSIONED ids and a pick is
                # unversioned. Matching on any other rule silently voids every arXiv paper --
                # the bug NR-52 recorded as C-14's tenth call site.
                resolved = {dedup_id(str(r["arxiv_id"])): r for r in got}
            except Exception as exc:  # noqa: BLE001 -- one bad case must not lose the rest
                print(f"  ! {case}: resolver failed: {type(exc).__name__}: {str(exc)[:60]}")
        for pid, paper, _arm in items:
            m = meta.get(pid) or resolved.get(dedup_id(pid)) or {}
            text = {
                **paper,
                "abstract": m.get("abstract", ""),
                "title": m.get("title") or paper.get("title", ""),
            }
            if not str(text.get("abstract") or "").strip():
                failed += 1
                continue
            try:
                second_verdict(case, ctx, text, DEFAULT_MODEL, cache_as=REPLICATE_TAG)
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ! {case}/{pid}: {type(exc).__name__}: {str(exc)[:60]}")
        print(f"  [{n}/{len(by_case)}] {case:<16} {done} drawn, {failed} void")
    print(f"\nreplicates drawn: {done}; void: {failed}")
    return 0


def report() -> int:
    orig = cached_sonnet()
    rep = cached_sonnet(REPLICATE_TAG)
    rows = [r for r in draw_sample() if son_of(rep, r[0], r[1]) is not None]
    a = [son_of(orig, c, p) for c, p, _pp, _arm in rows]
    b = [son_of(rep, c, p) for c, p, _pp, _arm in rows]
    n = len(rows)

    def flip(th: int) -> dict[str, Any]:
        f = sum(1 for x, y in zip(a, b, strict=True) if (x >= th) != (y >= th))
        rate = f / n if n else float("nan")
        se = math.sqrt(rate * (1 - rate) / n) if n else float("nan")
        return {
            "threshold": th,
            "flips": f,
            "n": n,
            "rate": round(rate, 4),
            "ci95": [round(max(0.0, rate - 1.96 * se), 4), round(min(1.0, rate + 1.96 * se), 4)],
            "implied_margin_sd_per_case": round(3 * math.sqrt(9 * rate * (1 - rate)), 2),
        }

    exact = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n if n else float("nan")
    k_bin = cohens_kappa([1 if x >= 2 else 0 for x in a], [1 if y >= 2 else 0 for y in b])
    k_quad = quadratic_kappa(list(a), list(b))
    f2, f1 = flip(2), flip(1)

    by_score: dict[str, dict[str, Any]] = {}
    for s in (0, 1, 2, 3):
        idx = [i for i, x in enumerate(a) if x == s]
        if idx:
            moved = sum(1 for i in idx if b[i] != s)
            by_score[str(s)] = {"n": len(idx), "changed": moved, "rate": round(moved / len(idx), 3)}

    verdict = {
        "kappa_binary_at_2": round(k_bin, 4),
        "sonnet_only_flip_rate": f2["rate"],
        "passes": bool(k_bin >= PASS_KAPPA and f2["rate"] <= PASS_FLIP),
        "kills": bool(k_bin < KILL_KAPPA or f2["rate"] >= KILL_FLIP),
    }
    verdict["grey"] = not verdict["passes"] and not verdict["kills"]

    out = {
        "_comment": (
            "NR-53. How much of NR-52's cross-judge disagreement is Sonnet disagreeing with "
            "ITSELF? The Claude path sends no temperature, so the Anthropic default 1.0 "
            "applies and every Sonnet verdict is a sample, while the GPT judge runs greedy at "
            "temperature 0. Run BEFORE fixing that, because at temperature 0 self-agreement is "
            "trivially ~1.0 and the quantity is unrecoverable. Bars pre-registered in the "
            "module docstring before any replicate was drawn. Derived by "
            "evals/sonnet_self_agreement.py; pinned by tests/test_sonnet_self_agreement.py."
        ),
        "pre_registered": {
            "pass": {"kappa_at_least": PASS_KAPPA, "flip_rate_at_most": PASS_FLIP},
            "kill": {"kappa_below": KILL_KAPPA, "flip_rate_at_least": KILL_FLIP},
            "written_before_any_replicate": True,
            "statistic_is_flip_rate_not_kappa": (
                "NR-52's labels are thresholds on the score, so a 2->3 redraw changes nothing "
                "while a 1->2 redraw flips sonnet-only and not consensus. Flip rate is what "
                "propagates into a margin; kappa is reported because it is what is quoted."
            ),
        },
        "model": DEFAULT_MODEL,
        "temperature": "unset -> Anthropic default 1.0 (the setting NR-52 used)",
        "n_replicated": n,
        "exact_score_agreement": round(exact, 4),
        "kappa_binary_at_2": round(k_bin, 4),
        "kappa_quadratic_0_3": round(k_quad, 4),
        "label_flips": {"sonnet_only_ge2": f2, "consensus_ge1": f1},
        "movement_by_original_score": by_score,
        "reference_points": {
            "gpt_sonnet_kappa_on_band": 0.199,
            "sonnet_only_margin_nr52": -3.41,
            "_comment": (
                "If self-agreement kappa is near the 0.199 measured BETWEEN judges, then what "
                "NR-52 called a judge difference is largely one judge's sampling noise."
            ),
        },
        "verdict": verdict,
    }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"replicated {n} papers at the settings NR-52 used (temperature unset -> 1.0)\n")
    print(f"  exact score agreement : {exact:.3f}")
    print(f"  kappa, binary at >=2  : {k_bin:.4f}   (GPT-vs-Sonnet on the band was 0.199)")
    print(f"  kappa, quadratic 0-3  : {k_quad:.4f}")
    print(
        f"\n  sonnet-only label flips (>=2): {f2['flips']}/{n} = {f2['rate']:.1%} "
        f"CI [{f2['ci95'][0]:.1%}, {f2['ci95'][1]:.1%}]"
    )
    print(
        f"  consensus label flips  (>=1): {f1['flips']}/{n} = {f1['rate']:.1%} "
        f"CI [{f1['ci95'][0]:.1%}, {f1['ci95'][1]:.1%}]"
    )
    print(
        f"\n  implied per-case sd injected into a margin: "
        f"{f2['implied_margin_sd_per_case']:.2f} net@2 (NR-52's sonnet-only margin: -3.41)"
    )
    print(f"  movement by original score: {by_score}")
    band = "PASS" if verdict["passes"] else ("KILL" if verdict["kills"] else "GREY")
    print(
        f"\nPRE-REGISTERED: pass k>={PASS_KAPPA} and flip<={PASS_FLIP:.0%}; "
        f"kill k<{KILL_KAPPA} or flip>={KILL_FLIP:.0%}  ->  {band}"
    )
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
