"""Restate the headline against every comparator we now have, and both shipped-source arms.

The paper's headline names ONE comparator -- Claude Opus 4.8, the v1 prompt, a 12-turn cap --
and reports **+3.88** against it. That figure is correct for the system it names, and this
script does not touch it. What it adds is the rest of the ladder, because two more
comparators exist now and a reader cannot tell from a single number how sensitive the margin
is to the rival's strength.

    Opus 4.8, v1, 12 turns   +1.84   (published; C-25 corrected from +1.56)
    Opus 4.8, v2, 30 turns   +2.16   same model, better harness
    Opus 5,   v2, 30 turns   +4.20   better model, same harness

**The decomposition is the finding.** Going from the published harness to the current one on
the SAME model buys the comparator +0.32. Swapping the model buys +2.04. So the published
comparator was not weak because we under-resourced it -- it was a fair instantiation of Opus
4.8, and Opus 4.8 is what it was. That distinction matters: "we gave the baseline a worse
prompt" is a methodological criticism, and it is not what happened.

Every cell is computed by the same two shared helpers the published figures use
(`band_testbeds.sign_test`, `bigram_report.paired_bootstrap`), not by a private
reimplementation, so a number here is comparable to a number in the paper. The C-25
correction is applied by RESTORING the forfeited picks (derived from `restated_runs.json`,
itself derived), never by hard-coding +1.84.

Both source arms are reported side by side throughout. arXiv-only is what `rr init
--measured` ships; arXiv+EPMC scores higher and is not shipped, and C-34 showed three
quarters of its core-25 gain is arXiv displacement rather than papers Europe PMC supplied.
Presenting one without the other would hide a live decision behind a chosen number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from band_testbeds import sign_test  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402

RES = EVALS / "results"
HEADLINE = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T225831Z.json"
ARMS = {
    # The published run. Kept as the anchor: the fresh control below has to reproduce it,
    # and it is the file every published denominator was computed from.
    "published_headline": HEADLINE,
    # The same-day matched pair. These two differ in `sources` and nothing else (P24), which
    # is what makes the arXiv+EPMC column readable as an effect rather than a redraw.
    "arxiv": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json",
    "arxiv_epmc": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json",
}
SPREADS = {
    "opus48_v2_30": "gold_spread_v2.json",
    "opus5_v2_30": "gold_spread_v2_opus5.json",
}
PUBLISHED_COMPARATOR = "opus48_v1_12"


def pt(score: Any) -> int:
    return 1 if int(score) >= 2 else -2


def net(scores) -> int:
    return sum(pt(s) for s in scores)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def load_run(fname: str) -> tuple[dict, dict]:
    """(RepoRadar top picks, baseline column) as judge scores, per case."""
    run = json.loads((RES / fname).read_text(encoding="utf-8"))
    rr = {
        e["case"]: [int(p["judge_score"]) for p in e["returned"]["reporadar_toppicks"]] for e in run
    }
    base = {e["case"]: [int(p["judge_score"]) for p in e["returned"]["baseline"]] for e in run}
    return rr, base


def load_spread(fname: str, draw: str = "1") -> dict[str, list[int]]:
    """One draw of a `gold_spread` sweep, as judge scores per case.

    Hallucinated and unjudgeable picks carry no verdict and are therefore ABSENT rather than
    scored negative -- the void-not-null rule. A row the model was never asked (`throttled`,
    `no_cli_login`) is not a measurement and is dropped, not counted as an abstention.
    """
    rows = json.loads((EVALS / fname).read_text(encoding="utf-8"))["results"]
    out = {}
    for key, row in rows.items():
        d, case = key.split("/", 1)
        if d == draw and row.get("status") == "ok":
            out[case] = list((row.get("scores") or {}).values())
    return out


# ── the comparators ───────────────────────────────────────────────────────────────────────
rr_by_arm, base_by_arm = {}, {}
for label, fname in ARMS.items():
    rr_by_arm[label], base_by_arm[label] = load_run(fname)

core = sorted(c for c in rr_by_arm["published_headline"] if not c.startswith(("bio-", "mat-")))

# The published comparator, restored. C-25: three baseline caches lost their transcripts to a
# 30-turn re-run and scored 0 returned papers while the same file's `ids` field still held
# their picks. `restated_runs.json` derives which caches and what the picks were worth; this
# reads that rather than restating +1.84 as a constant, so a re-run or a changed verdict moves
# this number instead of contradicting it.
restated = json.loads((EVALS / "restated_runs.json").read_text(encoding="utf-8"))
headline_row = next(r for r in restated["runs"] if r.get("is_headline"))
forfeited = {f["case"]: [int(s) for s in f["judge_scores"]] for f in headline_row["forfeited"]}
published = {c: list(base_by_arm["published_headline"][c]) for c in core}
for case, scores in forfeited.items():
    published[case] = published[case] + scores

comparators: dict[str, dict[str, list[int]]] = {PUBLISHED_COMPARATOR: published}
for label, fname in SPREADS.items():
    d = load_spread(fname)
    comparators[label] = {c: d[c] for c in core if c in d}

META = {
    PUBLISHED_COMPARATOR: {
        "model": "claude-opus-4-8",
        "prompt_version": "v1",
        "max_turns": 12,
        "source": f"baseline column of {HEADLINE}, C-25 restored",
        "is_published_comparator": True,
    },
    "opus48_v2_30": {
        "model": "claude-opus-4-8",
        "prompt_version": "v2",
        "max_turns": 30,
        "source": "gold_spread_v2.json draw 1",
        "is_published_comparator": False,
    },
    "opus5_v2_30": {
        "model": "claude-opus-5",
        "prompt_version": "v2",
        "max_turns": 30,
        "source": "gold_spread_v2_opus5.json draw 1",
        "is_published_comparator": False,
    },
}


def describe(picks: dict[str, list[int]]) -> dict:
    flat = [s for v in picks.values() for s in v]
    return {
        "n_cases": len(picks),
        "mean_net2": round(mean(net(v) for v in picks.values()), 2),
        "papers_per_case": round(mean(len(v) for v in picks.values()), 1),
        "shown": len(flat),
        "actionable": sum(1 for s in flat if s >= 2),
        "precision": round(sum(1 for s in flat if s >= 2) / len(flat), 3) if flat else None,
        "abstentions": sum(1 for v in picks.values() if not v),
    }


out: dict[str, Any] = {
    "_comment": (
        "The headline restated against every comparator now available, and both source arms, "
        "on the benchmark25 cohort. Derived by evals/restate_comparator.py; pinned by "
        "tests/test_comparator_ladder.py. The published +3.88 is the "
        "published_headline/opus48_v1_12 cell and is unchanged -- the ladder ADDS rows, it "
        "does not replace one. CIs from bigram_report.paired_bootstrap and sign tests from "
        "band_testbeds.sign_test, the same helpers every published figure uses."
    ),
    "cohort": "benchmark25",
    "n_cases": len(core),
    "comparators": {
        label: {**META[label], **describe(picks)} for label, picks in comparators.items()
    },
    "reporadar_arms": {},
    "paired": {},
}

for label, fname in ARMS.items():
    run = json.loads((RES / fname).read_text(encoding="utf-8"))
    e0 = next(e for e in run if e["case"] in core)
    out["reporadar_arms"][label] = {
        "run_file": fname,
        "sources": e0["sources"],
        "digest_window": e0["digest_window"],
        "w_embedding": e0["w_embedding"],
        "is_shipped_configuration": e0["sources"] == ["arxiv"],
        **describe({c: rr_by_arm[label][c] for c in core}),
    }

for arm in ARMS:
    out["paired"][arm] = {}
    for comp, picks in comparators.items():
        shared = sorted(c for c in core if c in rr_by_arm[arm] and c in picks)
        deltas = [float(net(rr_by_arm[arm][c]) - net(picks[c])) for c in shared]
        lo, hi = paired_bootstrap(deltas)
        st = sign_test(deltas)
        out["paired"][arm][comp] = {
            "n_cases": len(shared),
            "paired_delta": round(mean(deltas), 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "wins": st["pos"],
            "losses": st["neg"],
            "ties": st["ties"],
            "sign_p": round(st["p"], 6),
            "significant_at_05": st["p"] < 0.05,
        }

# ── what strengthening the comparator is actually made of ─────────────────────────────────
c = out["comparators"]
out["comparator_decomposition"] = {
    "_comment": (
        "Two steps between the published comparator and the current one. The harness step "
        "holds the model and changes the prompt and the turn cap; the model step holds the "
        "harness. Nearly all of the strengthening is the model, which is why the published "
        "figure is not a case of an under-resourced baseline."
    ),
    "published": c[PUBLISHED_COMPARATOR]["mean_net2"],
    "harness_v1_12_to_v2_30": round(
        c["opus48_v2_30"]["mean_net2"] - c[PUBLISHED_COMPARATOR]["mean_net2"], 2
    ),
    "model_opus48_to_opus5": round(
        c["opus5_v2_30"]["mean_net2"] - c["opus48_v2_30"]["mean_net2"], 2
    ),
    "total": round(c["opus5_v2_30"]["mean_net2"] - c[PUBLISHED_COMPARATOR]["mean_net2"], 2),
}

# The published cell, asserted rather than assumed: if this stops reproducing, the ladder is
# being computed differently from the paper and every other cell is suspect too.
pub = out["paired"]["published_headline"][PUBLISHED_COMPARATOR]
out["reproduces_published"] = {
    "expected_paired": headline_row["corrected"]["paired"],
    "expected_reporadar": headline_row["corrected"]["reporadar"],
    "expected_baseline": headline_row["corrected"]["baseline"],
    "got_paired": pub["paired_delta"],
    "got_reporadar": out["reporadar_arms"]["published_headline"]["mean_net2"],
    "got_baseline": c[PUBLISHED_COMPARATOR]["mean_net2"],
    "matches": (
        pub["paired_delta"] == headline_row["corrected"]["paired"]
        and out["reporadar_arms"]["published_headline"]["mean_net2"]
        == headline_row["corrected"]["reporadar"]
        and c[PUBLISHED_COMPARATOR]["mean_net2"] == headline_row["corrected"]["baseline"]
    ),
}

# The Opus 5 row is the only comparator that covers the scientific cases, so the 37-case
# figure is recorded apart rather than blended into a cohort no published denominator uses.
o5_all = load_spread(SPREADS["opus5_v2_30"])
all37 = sorted(set(o5_all) & set(rr_by_arm["arxiv_epmc"]))
out["all37_opus5"] = {}
for arm in ("arxiv", "arxiv_epmc"):
    deltas = [float(net(rr_by_arm[arm][c]) - net(o5_all[c])) for c in all37]
    lo, hi = paired_bootstrap(deltas)
    st = sign_test(deltas)
    out["all37_opus5"][arm] = {
        "n_cases": len(all37),
        "reporadar_mean": round(mean(net(rr_by_arm[arm][c]) for c in all37), 2),
        "opus5_mean": round(mean(net(o5_all[c]) for c in all37), 2),
        "paired_delta": round(mean(deltas), 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "wins": st["pos"],
        "losses": st["neg"],
        "ties": st["ties"],
        "sign_p": round(st["p"], 6),
        "significant_at_05": st["p"] < 0.05,
    }

(EVALS / "comparator_ladder.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

print(f"cohort: benchmark25, {len(core)} cases\n")
print(f"{'comparator':<26}{'net@2':>8}{'/case':>7}{'prec':>7}{'abst':>6}")
for label, v in out["comparators"].items():
    print(
        f"{label:<26}{v['mean_net2']:>+8.2f}{v['papers_per_case']:>7.1f}"
        f"{v['precision']:>7.3f}{v['abstentions']:>6}"
    )
print(f"\n{'RepoRadar arm':<26}{'net@2':>8}{'/case':>7}{'prec':>7}  sources")
for label, v in out["reporadar_arms"].items():
    print(
        f"{label:<26}{v['mean_net2']:>+8.2f}{v['papers_per_case']:>7.1f}"
        f"{v['precision']:>7.3f}  {','.join(v['sources'])}"
    )
print("\npaired (RepoRadar minus comparator):")
for arm, cells in out["paired"].items():
    for comp, v in cells.items():
        flag = "  *" if v["significant_at_05"] else "   "
        print(
            f"  {arm:<20} vs {comp:<16} {v['paired_delta']:>+6.2f} "
            f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  "
            f"{v['wins']}w/{v['losses']}l/{v['ties']}t  p={v['sign_p']:.4f}{flag}"
        )
print(f"\ndecomposition: {json.dumps(out['comparator_decomposition'], indent=1)}")
print(f"reproduces published: {out['reproduces_published']['matches']}")
print(f"\nall 37 vs Opus 5: {json.dumps(out['all37_opus5'], indent=1)}")
