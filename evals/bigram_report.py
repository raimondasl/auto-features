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

from ablation_report import (  # noqa: E402
    digest_width,
    load_arm,
    pool_mode,
    sign_test,
    summarise,
)

# evals/noise_floor.py, paired within one session. Which floor applies depends on how the
# arms got their candidates, and using the wrong one is a live way to overstate a result:
# a frozen-pool arm compared against the live floor calls a real effect unresolvable.
# The floor is a property of the whole MEASUREMENT CONFIGURATION, not of one axis of it.
# `mre_for` originally keyed on pool provenance alone, and widening the returned set on
# 2026-08-15 showed that was one dimension short: more papers per case means more chances
# for temperature-0 jitter in the gate and the rescore to move a paper across the display
# threshold, and each such paper is worth +1 or -2. The instrument got LESS sensitive as
# the digest got wider, which is the opposite of the usual more-data intuition.
MRE_PAIRED = 1.04  # live collection, window 10, three draws
MRE_FROZEN = 0.48  # one pool reused, window 10, two passes
MRE_FROZEN_W15 = 0.74  # one pool reused, window 15, two passes (residual sd 0.61 -> 0.93)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260812


def mre_for(provenance: str, width: str = "10") -> tuple[float, str]:
    """The floor that applies to arms with this provenance and digest width, and why.

    Derived rather than passed in. A flag would let a frozen comparison be read against the
    live floor by omission — the same shape as every other silently-wrong default this
    project has paid for.

    *width* was added on 2026-08-15 after measuring the floor at the new returned-set cut:
    **0.48 at window 10, 0.74 at window 15** on the same frozen pool. Keying on provenance
    alone had this function confidently returning a floor 35% too tight for every future
    window-15 experiment — a guard that is precise about one dimension and silent about
    another reads as authority on both.

    An unrecognised width falls back to the widest floor measured rather than the nearest,
    because under-reporting the floor turns noise into a finding, and that is the direction
    that costs a published claim.
    """
    if not provenance.startswith("frozen"):
        # The live branch was width-BLIND until 2026-08-15 and returned the label
        # "window 10" whatever the arm's actual width — so the C-8 fix covered the frozen
        # path and left this one committing the identical error. It surfaced on the
        # scan-source arms, a live window-15 comparison read against the window-10 live
        # floor. The live floor at 15 has never been measured; the frozen one rose 0.48 ->
        # 0.74 (x1.54) when the cut widened, so 1.04 is a LOWER BOUND here, and under-
        # reporting a floor turns noise into a finding. Said out loud rather than scaled:
        # inventing 1.60 would be an unmeasured number wearing a measurement's authority.
        if width == "10":
            return MRE_PAIRED, "live collection, window 10"
        return MRE_PAIRED, (
            f"live collection, window {width} — UNMEASURED; {MRE_PAIRED} is the window-10 "
            "value and a LOWER BOUND"
        )
    if width == "10":
        return MRE_FROZEN, "frozen pool, reused, window 10"
    if width == "15":
        return MRE_FROZEN_W15, "frozen pool, reused, window 15"
    return MRE_FROZEN_W15, f"frozen pool, reused, window {width} — UNMEASURED, widest known"


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


def check_labels(label: str, arm: dict[str, dict[str, Any]], field: str = "bigram_mode") -> None:
    """The run files record their own arm; a mislabelled file is a silent swap.

    *field* names the result key that identifies the arm, so this serves any single-flag
    experiment — `bigram_mode` for the phrase-query arms, `absent_category` for the ranker
    ones, `gate_depth` for the triage-depth ones. Files differing by one flag are easy to
    pass in the wrong order.

    Recorded values are compared as strings because a command-line label always is one and
    the arm it names may not be: `gate_depth` is an int. Comparing raw would make the guard
    fire on every numeric arm — a check that always fails gets deleted, which is worse than
    one that never does.

    **Boolean arms** get one extra convention, because a bool field would otherwise force
    every report to label its arms `True` and `False`. For a boolean `field`, the label
    may also be the field's own name (the True arm) or `no-<field>` / `no_<field>` (the
    False arm) — so `hybrid` vs `no-hybrid` reads as intended and still refuses a genuine
    swap, since labelling the False arm `hybrid` resolves to True and fails. Added when
    the fusion ablation hit exactly this: readable labels the guard could not accept.
    """
    recorded = {r.get(field) for r in arm.values()}
    if recorded == {None}:
        print(f"  ! {label}: no `{field}` recorded (run predates the flag) — trusting label")
        return
    if recorded <= {True, False}:
        negated = label.lower() in (f"no-{field}", f"no_{field}")
        claimed = False if negated else (True if label.lower() == field else None)
        if claimed is not None:
            if recorded != {claimed}:
                raise SystemExit(
                    f"arm labelled {label!r} means {field}={claimed} but its run file "
                    f"records {field}={sorted(map(str, recorded))} — refusing"
                )
            return
    if {str(v) for v in recorded} != {str(label)}:
        raise SystemExit(
            f"arm labelled {label!r} contains {field}={sorted(map(str, recorded))} — "
            "refusing to report an arm under a name its own run file contradicts"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", metavar="LABEL=FILE", help="control arm first")
    ap.add_argument("--out", default="evals/.work/bigrams.json")
    ap.add_argument(
        "--across-windows",
        action="store_true",
        help="Permit arms with different `digest_window` values. Refused by default: the "
        "returned-set width moved 10 -> 15 on 2026-08-15 and is worth +1.24 net@2/case by "
        "itself, so mixing widths measures the width and reports it under the arms' names. "
        "The one experiment this is for is the width experiment.",
    )
    ap.add_argument(
        "--label-field",
        default="bigram_mode",
        help="Result key identifying the arm, so a file cannot be reported under a name it "
        "contradicts. `bigram_mode` for the phrase-query arms, `absent_category` for the "
        "ranker ones.",
    )
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
        # `frozen-seeded` (the arm that collected the pool and saved it) and `frozen:<fp>`
        # (an arm that reused it) are different *histories* and `provenance` rightly says
        # so — the seeding run did collect live. But comparability asks a narrower
        # question: did these arms rank the same candidates? The seeding run ranks exactly
        # what it wrote, so when every shared case has an identical fingerprint the answer
        # is yes, and `same_pool` is the check that decides it.
        #
        # Without this the natural shape of a frozen experiment — arm 1 seeds, arm 2
        # reuses — is unreportable, and every such pair needs a throwaway collection first.
        # Found on 2026-08-16 by the w_embedding arms: refused despite identical
        # fingerprints and identical pool sizes on all 25 cases.
        #
        # The absent-category experiment hit this in August and paid the throwaway pass,
        # noting the refusal was right. It was right *then*: there was no fingerprint check
        # at this decision point, so mode equality was the only evidence available. The
        # workaround was sufficient, not necessary.
        #
        # What makes the relaxation safe is that the one way a seeding arm and a reuse arm
        # can rank different candidates is an EMPTY pool: `save_frozen_pool` refuses to
        # store one (an empty pool and a failed collection are the same bytes), so the
        # reuse arm re-collects that case live and reports `frozen-seeded` for it — which
        # makes the run's own modes mixed, and `provenance` already returns "mixed" and is
        # refused above. So the divergence this check would have caught is caught anyway.
        from noise_floor import same_pool

        families = {m.split(":")[0].replace("frozen-seeded", "frozen") for m in modes.values()}
        differing = [
            case for label in labels[1:] for case in same_pool(arms[labels[0]], arms[label])
        ]
        if families != {"frozen"} or differing:
            raise SystemExit(
                "refusing to compare arms with different pool provenance: "
                + ", ".join(f"{k}={v}" for k, v in modes.items())
                + (
                    f"\n  cases whose frozen pools differ: {sorted(set(differing))}"
                    if differing
                    else ""
                )
            )
        print(
            "  note: one arm seeded the frozen pool and the other reused it; every shared "
            "case has an identical fingerprint, so they rank the same candidates."
        )
    # The same refusal for the returned-set width. Widening it 10 -> 15 was worth +1.24
    # net@2/case — bigger than any treatment effect published here — so a mixed-width
    # comparison measures the width and attributes it to whatever the arms were named.
    widths = {label: digest_width(arms[label]) for label in labels}
    if len(set(widths.values())) > 1:
        if not args.across_windows:
            raise SystemExit(
                "refusing to compare arms with different digest windows: "
                + ", ".join(f"{k}={v}" for k, v in widths.items())
                + " — widening 10 to 15 is worth +1.24 net@2/case on its own. Pass "
                "--across-windows if the WIDTH is the treatment under test."
            )
        # Loud, every time, like noise_floor's --assume-unlabelled-live. An experiment
        # whose treatment IS the width has to say so out loud rather than inherit silence.
        print(
            "  ! --across-windows: arms differ in returned-set width ("
            + ", ".join(f"{k}={v}" for k, v in widths.items())
            + "). Valid only if the width is what is being measured."
        )
    for label in labels:
        check_labels(label, arms[label], args.label_field)
    # Derived from how the arms were collected, never passed in — see `mre_for`.
    # The WIDEST floor across the arms, not the control's. A mixed-width comparison
    # (permitted only under --across-windows) has the noisier arm's variance in its
    # difference, so reading it against the narrower arm's floor would understate what the
    # instrument can see — the exact direction that turns noise into a finding.
    floors = [mre_for(modes[label], widths[label]) for label in labels]
    mre, floor_why = max(floors, key=lambda pair: pair[0])

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

    print(f"\npaired against {control!r}, same session (MRE = {mre:.2f} net@2/case — {floor_why}):")
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
        # Magnitude only. Whether THIS draw established it is the CI's question, printed
        # beside it — source_ab_report.py conflated the two once and overstated a result.
        resolvable = "past the floor" if abs(mean) >= mre else "inside the floor"
        # "Past the floor" against a floor that is only a lower bound is not a finding:
        # the true floor may be above the effect. "Inside" stays sound, because inside a
        # lower bound is inside anything larger.
        if resolvable == "past the floor" and "LOWER BOUND" in floor_why:
            resolvable = "past a LOWER-BOUND floor — NOT resolved"
        paired[label] = {"mean": mean, "ci": [lo, hi], "sign_p": p, "resolvable": resolvable}
        print(
            f"  {label:>10}  {mean:+6.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
            f"{pos}+/{neg}-/{ties}=  sign p = {p:.4f}   {resolvable}"
        )

    print("\n" + "-" * 78)
    for label in labels[1:]:
        if not divs[label]["changed_cases"]:
            print(f"{label}: VOID. The arm did not change retrieval; nothing was measured.")
        elif paired[label]["resolvable"] == "past the floor":
            lo, hi = paired[label]["ci"]
            clear = lo > 0 or hi < 0
            shown = "and the CI excludes 0" if clear else "but the CI spans 0"
            verdict = "established" if clear else "suggestive, not established"
            print(
                f"{label}: {paired[label]['mean']:+.2f}/case is past the {mre} floor "
                f"{shown} — {verdict}."
            )
        else:
            print(
                f"{label}: {paired[label]['mean']:+.2f}/case is INSIDE the {mre} floor — "
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
