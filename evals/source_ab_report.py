"""Two-arm report for adding a paper source, with the negative controls read separately.

    uv run python evals/source_ab_report.py arxiv=judge-...A.json +s2=judge-...B.json

Built for the Semantic Scholar A/B — the experiment `evals/RESULTS.md` finding 3 only
appeared to run, because the arm it measured sent malformed queries and S2 answered with
nothing (C-9). Two things make this report different from a mean.

**Arm validity.** The stage-1 probe showed S2 papers reaching ranked top-10s, but "the
channel delivered" has to be re-checked in the judged run itself, not assumed from a
cheaper measurement made under different settings. Papers whose ids the control never saw
are counted per case; if the treatment shows none, the arm is VOID and its delta means
nothing. Three findings have already been lost to that shape.

**The negative controls are reported apart from the mean, and this is the point.** The
benchmark labels `webdev`, `cli` and `http` negative controls with `gold_n: 0` — but that
encodes "no gold *arXiv* papers", a claim about coverage rather than about whether research
exists that could improve a web framework or an HTTP client. TLS and connection-pool policy,
retry/backoff, certificate validation, session security: real literature, largely in
USENIX/CCS/WWW, which S2 indexes and arXiv does not.

Tier B does not beg the question — `negative_control` is read only by `run_eval.py` (Tier A,
offline fixtures), so the judge scores every paper on merit with no knowledge of the label.
That makes the controls the most informative cases here rather than merely the riskiest:

* judge scores **2-3** on the new papers → the label is arXiv-specific, not true, and S2 is
  filling a genuine coverage gap;
* judge scores **0-1** → the ranker is admitting topically loose papers, and the fix is a
  ranker change (penalise absent categories for uncategorised *sources*) rather than a
  verdict on S2.

Averaging those three cases into 25 hides both readings, so this prints the judge's score
distribution on exactly the papers the treatment added.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_report import load_arm, pool_mode, sign_test, summarise  # noqa: E402
from bigram_report import MRE_PAIRED, paired_bootstrap, top10_ids  # noqa: E402

# `benchmark.yaml` marks these `negative_control: true`. Read from the benchmark rather
# than hardcoded, so adding a control cannot silently leave it pooled into the mean.
BENCHMARK = Path(__file__).resolve().parent / "benchmark.yaml"


def negative_controls() -> set[str]:
    import yaml

    bench = yaml.safe_load(BENCHMARK.read_text(encoding="utf-8"))
    return {c["name"] for c in bench["cases"] if c.get("negative_control")}


def returned_records(record: dict[str, Any], key: str = "reporadar_top10") -> list[dict[str, Any]]:
    return (record.get("returned") or {}).get(key) or []


def added_papers(control: dict[str, Any], treat: dict[str, Any]) -> list[dict[str, Any]]:
    """Papers the treatment returned that the control never did — the arm's actual effect."""
    seen = top10_ids(control)
    return [r for r in returned_records(treat) if r.get("arxiv_id") and r["arxiv_id"] not in seen]


def score_histogram(records: list[dict[str, Any]]) -> Counter[Any]:
    return Counter(r.get("judge_score", r.get("score")) for r in records)


def source_marked_ids(arm: dict[str, dict[str, Any]], prefix: str) -> int:
    """How many returned papers carry a non-arXiv id prefix (`ss:`, `dblp:`, `iacr:`)."""
    return sum(
        1
        for record in arm.values()
        for r in returned_records(record)
        if str(r.get("arxiv_id", "")).startswith(prefix)
    )


def check_arms(
    control: dict[str, dict[str, Any]],
    treat: dict[str, dict[str, Any]],
    prefix: str,
) -> None:
    """Verify the two files really are control and treatment, from their CONTENT.

    Run files do not record which `--sources` produced them, so a label passed on the
    command line is an unverified claim, and two files differing by one flag are easy to
    pass in the wrong order. The papers themselves are not a claim: a source that only the
    treatment enabled stamps its own id prefix on every paper it contributes, so the
    control must carry none and the treatment must carry some.

    This is a stronger guard than the `bigram_mode` label check in `bigram_report`, which
    can only detect a *recorded* mismatch. Here a swap is detected even if nobody recorded
    anything.
    """
    in_control = source_marked_ids(control, prefix)
    in_treat = source_marked_ids(treat, prefix)
    if in_control:
        raise SystemExit(
            f"the CONTROL arm contains {in_control} {prefix!r} paper(s) — the arms are "
            "swapped, or the control was run with the source enabled. Refusing to report."
        )
    if not in_treat:
        # Not fatal: the source may have contributed papers that carry real arXiv ids, and
        # the caller still needs the added-paper counts below to judge validity.
        print(
            f"  ! the treatment arm returned no {prefix!r} papers. Either the source "
            "contributed only papers that also exist on arXiv, or the arm did not run — "
            "read ARM VALIDITY below before believing any delta."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs=2, metavar="LABEL=FILE", help="control first, treatment second")
    ap.add_argument("--out", default="evals/.work/source_ab.json")
    ap.add_argument(
        "--id-prefix",
        default="ss:",
        help="Synthetic id prefix the treatment's source stamps on its papers "
        "(ss: for Semantic Scholar, dblp:, iacr:). Used to verify the arms from their "
        "content rather than from the labels given on the command line.",
    )
    args = ap.parse_args()

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.arms:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=FILE, got {spec!r}")
        arms[label] = load_arm(path)
    (control_label, control), (treat_label, treat) = arms.items()

    if set(control) != set(treat):
        raise SystemExit(f"arms cover different cases: {sorted(set(control) ^ set(treat))}")
    modes = {control_label: pool_mode(control), treat_label: pool_mode(treat)}
    if len(set(modes.values())) > 1:
        raise SystemExit(
            "refusing to compare arms with different pool provenance: "
            + ", ".join(f"{k}={v}" for k, v in modes.items())
        )

    controls = negative_controls() & set(control)
    ordinary = sorted(set(control) - controls)
    cases = sorted(control)

    print("=" * 78)
    print(f"SOURCE A/B — {len(cases)} cases, {control_label!r} vs {treat_label!r}")
    print(f"pool provenance: {modes[control_label]}")
    print("=" * 78)
    check_arms(control, treat, args.id_prefix)

    print(f"\n{'arm':>12} {'net@2':>7} {'shown':>6} {'act':>5} {'prec':>6} {'abst':>5} {'neg':>4}")
    summaries = {}
    for label, arm in ((control_label, control), (treat_label, treat)):
        s = summaries[label] = summarise(arm)
        prec = f"{s['precision']:.3f}" if s["precision"] is not None else "  n/a"
        print(
            f"{label:>12} {s['mean_net2']:+7.2f} {s['shown']:6} {s['actionable']:5} "
            f"{prec:>6} {s['abstained']:5} {s['net_negative']:4}"
        )

    # ---- arm validity -----------------------------------------------------------------
    print("\nARM VALIDITY — did the treatment actually return papers the control did not?")
    added = {c: added_papers(control[c], treat[c]) for c in cases}
    n_added = sum(len(v) for v in added.values())
    cases_with = sum(1 for v in added.values() if v)
    print(f"  {n_added} new papers returned, across {cases_with}/{len(cases)} cases")
    if n_added == 0:
        print("  VOID — the treatment returned nothing new. No delta below is a measurement.")

    # ---- the mean, and the same mean without the controls ------------------------------
    def deltas_for(subset: list[str]) -> list[float]:
        return [
            treat[c]["reporadar_toppicks"]["net_value@2"]
            - control[c]["reporadar_toppicks"]["net_value@2"]
            for c in subset
        ]

    print(f"\npaired, same session (MRE = {MRE_PAIRED:.2f} net@2/case):")
    reported = {}
    for name, subset in (
        ("all cases", cases),
        ("excluding controls", ordinary),
        ("controls only", sorted(controls)),
    ):
        if not subset:
            continue
        d = deltas_for(subset)
        mean = statistics.mean(d)
        lo, hi = paired_bootstrap(d)
        pos, neg, ties, p = sign_test(d)
        # Two separate questions, and conflating them overstates a result. The MRE asks
        # whether an effect this SIZE is detectable in principle; the interval and the sign
        # test ask whether THIS draw established one. A mean past the floor whose CI still
        # spans zero is "big enough to see, not yet shown" — not "resolved".
        big_enough = abs(mean) >= MRE_PAIRED
        # Strict: an interval whose bound sits AT zero does not exclude it. Written as
        # `(lo > 0) == (hi > 0)` this called [-2.14, +0.00] "excludes 0", because neither
        # bound is positive — a sign agreement test, not a containment test.
        excludes_zero = lo > 0 or hi < 0
        if big_enough and excludes_zero:
            verdict = "past the floor, CI excludes 0"
        elif big_enough:
            verdict = "past the floor, CI spans 0 — suggestive, not established"
        else:
            verdict = "inside the floor — unresolvable, not absent"
        reported[name] = {
            "n": len(subset),
            "mean": mean,
            "ci": [lo, hi],
            "sign_p": p,
            "verdict": verdict,
        }
        print(
            f"  {name:20} n={len(subset):2d}  {mean:+6.2f}  CI [{lo:+.2f}, {hi:+.2f}]  "
            f"{pos}+/{neg}-/{ties}=  p={p:.4f}"
        )
        print(f"  {'':20} {verdict}")

    # ---- the controls, in full ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("NEGATIVE CONTROLS — `gold_n: 0` means no gold ARXIV papers, which is a claim")
    print("about coverage. The judge never sees the label; it scores on merit.")
    print("-" * 78)
    print(f"\n{'case':10} {'ctrl':>6} {'treat':>6} {'added':>6}   judge scores on the added papers")
    control_detail = {}
    for case in sorted(controls):
        c_net = control[case]["reporadar_toppicks"]["net_value@2"]
        t_net = treat[case]["reporadar_toppicks"]["net_value@2"]
        hist = score_histogram(added[case])
        # `None` sorts last: it means the judge never scored that paper, which is a
        # different thing from a zero and must not be read as one.
        ordered = sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0]))
        shown = "  ".join(f"{s}:{n}" for s, n in ordered)
        control_detail[case] = {"control": c_net, "treatment": t_net, "scores": dict(hist)}
        print(f"{case:10} {c_net:+6.1f} {t_net:+6.1f} {len(added[case]):6d}   {shown or '-'}")

    control_scores = [
        s for c in controls for r in added[c] if isinstance(s := r.get("judge_score"), int)
    ]
    actionable = sum(1 for s in control_scores if s >= 2)
    loose = sum(1 for s in control_scores if s < 2)
    print(f"\n  added papers on controls judged ACTIONABLE (2-3): {actionable}")
    print(f"  added papers on controls judged loose      (0-1): {loose}")
    if actionable + loose:
        if actionable > loose:
            print("\n  => The 'negative control' label looks arXiv-specific, not true. S2 is")
            print("     surfacing work a judge considers genuinely applicable to these repos.")
        else:
            print("\n  => The controls are behaving as controls: the added papers are topically")
            print("     loose. If they reach digests, the fix is the ranker's treatment of")
            print("     uncategorised sources, not a verdict on S2.")

    print(f"\n{'case':11}{control_label:>12}{treat_label:>12}{'delta':>8}")
    for case in cases:
        c_net = control[case]["reporadar_toppicks"]["net_value@2"]
        t_net = treat[case]["reporadar_toppicks"]["net_value@2"]
        mark = " *" if case in controls else ""
        print(f"{case:11}{c_net:>12.1f}{t_net:>12.1f}{t_net - c_net:>+8.1f}{mark}")
    print("  * negative control")

    Path(args.out).write_text(
        json.dumps(
            {
                "control": control_label,
                "treatment": treat_label,
                "summaries": summaries,
                "paired": reported,
                "arm_validity": {"new_papers": n_added, "cases_with_new": cases_with},
                "negative_controls": control_detail,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
