"""Paired comparison of two triage repo-representations on the same labelled papers.

    uv run python evals/diagnose_triage.py --repo-context keywords
    uv run python evals/diagnose_triage.py --repo-context readme
    uv run python evals/compare_triage.py \
        evals/.work/diag_triage_keywords.json evals/.work/diag_triage_readme.json

**Report net@2, not accuracy.** The gate's errors are asymmetric — returning a junk paper
costs 2, missing a good one costs 1 — so accuracy answers a question nobody asked. Judged on
accuracy the README variant looked like a null (21 fixed, 17 broke, p = 0.63); judged on
net@2 the same runs are +16 (+31%), because most of its flips were false positives becoming
correct abstentions, each worth +2. See RESULTS.md -> "the README variant, re-judged".


Comparing two aggregate precision numbers cannot separate a real effect from Haiku's
non-determinism. This is paired per paper: for each one, did the variant FIX a baseline
mistake, BREAK a baseline success, or change nothing? A real improvement shows many more
fixes than breaks; noise shows roughly equal counts in both directions.

Also reports a two-sided exact binomial on the discordant pairs (McNemar), so "12 fixed,
9 broke" is not read as progress.
"""

import json
import math
import random
import sys
from pathlib import Path

ACTIONABLE = 2


def load(p: Path) -> dict[tuple[str, str], dict]:
    return {(r["case"], r["id"]): r for r in json.loads(p.read_text(encoding="utf-8"))}


def binom_two_sided(k: int, n: int) -> float:
    """P(|X - n/2| >= |k - n/2|) under X ~ Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    d = abs(k - n / 2)
    tot = 0.0
    for i in range(n + 1):
        if abs(i - n / 2) >= d - 1e-9:
            tot += math.comb(n, i)
    return min(1.0, tot / (2**n))


def main() -> int:
    base_p, var_p = Path(sys.argv[1]), Path(sys.argv[2])
    base, var = load(base_p), load(var_p)
    shared = sorted(set(base) & set(var))
    print(f"paired on {len(shared)} papers ({base_p.name} vs {var_p.name})\n")

    fixed = broke = same_ok = same_bad = 0
    per_case: dict[str, list[int]] = {}
    for key in shared:
        truth = base[key]["judge"] >= ACTIONABLE
        b_ok = (base[key]["triage"] >= ACTIONABLE) == truth
        v_ok = (var[key]["triage"] >= ACTIONABLE) == truth
        per_case.setdefault(key[0], [0, 0])
        if v_ok and not b_ok:
            fixed += 1
            per_case[key[0]][0] += 1
        elif b_ok and not v_ok:
            broke += 1
            per_case[key[0]][1] += 1
        elif b_ok:
            same_ok += 1
        else:
            same_bad += 1

    def acc(d: dict) -> tuple[float, float, float]:
        tp = sum(1 for k in shared if d[k]["triage"] >= ACTIONABLE and d[k]["judge"] >= ACTIONABLE)
        fp = sum(1 for k in shared if d[k]["triage"] >= ACTIONABLE and d[k]["judge"] < ACTIONABLE)
        fn = sum(1 for k in shared if d[k]["triage"] < ACTIONABLE and d[k]["judge"] >= ACTIONABLE)
        n = len(shared)
        correct = sum(
            1 for k in shared if (d[k]["triage"] >= ACTIONABLE) == (d[k]["judge"] >= ACTIONABLE)
        )
        return (
            tp / (tp + fp) if tp + fp else float("nan"),
            tp / (tp + fn) if tp + fn else float("nan"),
            correct / n,
        )

    bp, br, ba = acc(base)
    vp, vr, va = acc(var)
    print(f"{'':10} {'precision':>10} {'recall':>8} {'accuracy':>9}")
    print(f"{'baseline':10} {bp:10.2f} {br:8.2f} {ba:9.2f}")
    print(f"{'variant':10} {vp:10.2f} {vr:8.2f} {va:9.2f}")
    print(f"{'delta':10} {vp - bp:+10.2f} {vr - br:+8.2f} {va - ba:+9.2f}")

    n_disc = fixed + broke
    p = binom_two_sided(fixed, n_disc)
    print(
        f"\npaired outcome: {fixed} fixed, {broke} broke, "
        f"{same_ok} both right, {same_bad} both wrong"
    )
    print(f"discordant pairs: {n_disc}   two-sided exact binomial p = {p:.4f}")
    print("(accuracy view — treats a false positive and a false negative as equally bad)")

    # net@2 is what the product optimises, and its errors are ASYMMETRIC: returning a junk
    # paper costs 2, missing a good one costs 1. Judging a gate on accuracy therefore
    # answers a question nobody asked — it reported the README variant as a null (p=0.63)
    # while that variant was worth +16 net@2, because most of its flips were false
    # positives turning into correct abstentions, each worth +2.
    def contrib(r: dict) -> int:
        if r["triage"] < ACTIONABLE:
            return 0
        return 1 if r["judge"] >= ACTIONABLE else -2

    deltas = [contrib(var[k]) - contrib(base[k]) for k in shared]
    total = sum(deltas)
    rng = random.Random(7)
    boots = sorted(sum(rng.choice(deltas) for _ in deltas) for _ in range(4000))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    p_le0 = sum(1 for b in boots if b <= 0) / len(boots)
    b_net = sum(contrib(base[k]) for k in shared)
    v_net = sum(contrib(var[k]) for k in shared)
    print(f"\nnet@2   baseline {b_net:+d}   variant {v_net:+d}   delta {total:+d}")
    print(f"paired bootstrap 95% CI [{lo:+d}, {hi:+d}]   P(delta <= 0) = {p_le0:.3f}")
    print("(net@2 view — the metric the digest is scored on; use this one)")

    print(f"\n{'case':11} {'fixed':>6} {'broke':>6}")
    for case, (f, b) in sorted(per_case.items(), key=lambda kv: kv[1][1] - kv[1][0]):
        if f or b:
            print(f"{case:11} {f:6d} {b:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
