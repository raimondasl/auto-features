"""Three-arm report for the phrase-query experiment (`--rr-bigrams`).

    uv run python evals/bigram_report.py \
        adjacent=judge-...A.json verified=judge-...B.json none=judge-...C.json

`build_queries` pairs each keyword with its TF-IDF neighbour and sends the pair as a quoted
phrase, with nothing requiring the two words to belong together. Measured on real profiles
it emits ``"use page"`` for duckdb, ``"data cd"`` for redis, ``"server code"`` for ruff.
The three arms are that policy (``adjacent``), the same pairs restricted to phrases the
repository literally contains (``verified``), and no phrase queries at all (``none``).

**What this reports that a mean cannot.** Two failure modes have already cost this project
a published number, and both are checked here before any delta is believed:

* **A void arm.** The first IACR measurement scored two arms identically because *zero*
  IACR papers reached a top-10 — reported as "no effect" that would have been a
  manufactured null. So this prints the top-10 divergence between each arm and the
  control. An arm whose returned papers are identical everywhere did not run; it is VOID,
  and its delta of 0.00 means nothing.
* **A null read as an absence.** The benchmark's paired same-session minimum resolvable
  effect is **1.04 net@2/case** (`evals/noise_floor.py`). A delta inside that is
  *unresolvable*, not *absent* — the distinction I got wrong when sizing the IACR subset
  against headroom instead of a plausible effect. Every delta is printed against the floor.

No prediction is pre-registered for the direction of the effect, and that is deliberate:
`verified` removes phrases that cannot match anything, not phrases that are unhelpful. On
`db` it trades ``"use page"`` for ``"guide support"`` — both junk, but only one is
unmatchable. Whether that distinction reaches net@2 is the open question.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_report import load_arm, pool_mode, sign_test, summarise  # noqa: E402

# evals/noise_floor.py, three live draws, paired within one session.
MRE_PAIRED = 1.04
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260812


def top10_ids(record: dict[str, Any]) -> set[str]:
    """The papers this case actually returned, for arm-divergence."""
    returned = (record.get("returned") or {}).get("reporadar_top10") or []
    return {r.get("arxiv_id", "") for r in returned if r.get("arxiv_id")}


def divergence(control: dict[str, Any], arm: dict[str, Any], cases: list[str]) -> dict[str, Any]:
    """How differently the two arms retrieved, per case and overall.

    `changed_cases` is the number whose returned top-10 is not identical. Zero means the
    flag did nothing that reached the output, and no delta computed from it is a
    measurement of the flag.
    """
    changed, jaccards = 0, []
    for case in cases:
        a, b = top10_ids(control[case]), top10_ids(arm[case])
        if a != b:
            changed += 1
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 1.0)
    return {
        "changed_cases": changed,
        "of": len(cases),
        "mean_jaccard": statistics.mean(jaccards) if jaccards else 1.0,
    }


def paired_bootstrap(deltas: list[float], n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Percentile CI for the mean paired delta, resampling CASES (the unit of pairing)."""
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(BOOTSTRAP_SEED)
    k = len(deltas)
    means = sorted(statistics.mean([deltas[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def check_labels(label: str, arm: dict[str, dict[str, Any]]) -> None:
    """The run files record their own arm; a mislabelled file is a silent swap."""
    recorded = {r.get("bigram_mode") for r in arm.values()}
    if recorded == {None}:
        print(f"  ! {label}: no `bigram_mode` recorded (run predates the flag) — trusting label")
        return
    if recorded != {label}:
        raise SystemExit(
            f"arm labelled {label!r} contains bigram_mode={sorted(map(str, recorded))} — "
            "refusing to report an arm under a name its own run file contradicts"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", metavar="LABEL=FILE", help="control arm first")
    ap.add_argument("--out", default="evals/.work/bigrams.json")
    args = ap.parse_args()

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.arms:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=FILE, got {spec!r}")
        arms[label] = load_arm(path)

    labels = list(arms)
    control = labels[0]
    base = set(arms[control])
    for label in labels[1:]:
        if set(arms[label]) != base:
            raise SystemExit(
                f"arm {label!r} covers different cases: {sorted(base ^ set(arms[label]))}"
            )
    modes = {label: pool_mode(arms[label]) for label in labels}
    if len(set(modes.values())) > 1:
        raise SystemExit(
            "refusing to compare arms with different pool provenance: "
            + ", ".join(f"{k}={v}" for k, v in modes.items())
        )
    for label in labels:
        check_labels(label, arms[label])

    cases = sorted(base)
    print("=" * 78)
    print(f"PHRASE-QUERY ARMS — {len(cases)} cases, control = {control!r}, pool {modes[control]}")
    print("=" * 78)

    print(f"\n{'arm':>10} {'net@2':>7} {'shown':>6} {'act':>5} {'prec':>6} {'abst':>5} {'neg':>4}")
    summaries = {}
    for label in labels:
        s = summaries[label] = summarise(arms[label])
        prec = f"{s['precision']:.3f}" if s["precision"] is not None else "  n/a"
        print(
            f"{label:>10} {s['mean_net2']:+7.2f} {s['shown']:6} {s['actionable']:5} "
            f"{prec:>6} {s['abstained']:5} {s['net_negative']:4}"
        )

    print("\nARM VALIDITY — did the flag change what was returned?")
    divs = {}
    for label in labels[1:]:
        d = divs[label] = divergence(arms[control], arms[label], cases)
        verdict = (
            "VOID — identical output, no delta below is a measurement"
            if not d["changed_cases"]
            else "ok"
        )
        print(
            f"  {label:>10}  {d['changed_cases']}/{d['of']} cases changed, "
            f"mean top-10 Jaccard {d['mean_jaccard']:.2f}   {verdict}"
        )

    print(f"\n{'case':11}" + "".join(f"{label:>10}" for label in labels))
    for case in cases:
        row = "".join(
            f"{arms[label][case]['reporadar_toppicks']['net_value@2']:>10.1f}" for label in labels
        )
        print(f"{case:11}{row}")

    print(f"\npaired against {control!r}, same session (MRE = {MRE_PAIRED:.2f} net@2/case):")
    paired = {}
    for label in labels[1:]:
        deltas = [
            arms[label][c]["reporadar_toppicks"]["net_value@2"]
            - arms[control][c]["reporadar_toppicks"]["net_value@2"]
            for c in cases
        ]
        mean = statistics.mean(deltas)
        lo, hi = paired_bootstrap(deltas)
        pos, neg, ties, p = sign_test(deltas)
        resolvable = "RESOLVED" if abs(mean) >= MRE_PAIRED else "inside the floor"
        paired[label] = {"mean": mean, "ci": [lo, hi], "sign_p": p, "resolvable": resolvable}
        print(
            f"  {label:>10}  {mean:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
            f"{pos}+/{neg}-/{ties}=  sign p = {p:.4f}   {resolvable}"
        )

    print("\n" + "-" * 78)
    for label in labels[1:]:
        if not divs[label]["changed_cases"]:
            print(f"{label}: VOID. The arm did not change retrieval; nothing was measured.")
        elif paired[label]["resolvable"] == "RESOLVED":
            print(f"{label}: effect {paired[label]['mean']:+.2f}/case, outside the noise floor.")
        else:
            print(
                f"{label}: {paired[label]['mean']:+.2f}/case is INSIDE the {MRE_PAIRED} floor — "
                "unresolvable at this n, which is not the same as absent."
            )

    Path(args.out).write_text(
        json.dumps(
            {"control": control, "summaries": summaries, "paired": paired, "divergence": divs},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
