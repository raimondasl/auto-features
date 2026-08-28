"""Freeze P26: the completed 37-case Opus 5 comparator arm. Run files are gitignored.

Opus 5 draw 1 now covers all 37 cases (25 core + 6 bio + 6 materials). Every figure this
artifact holds is derived from `gold_spread_v2_opus5.json`, which IS in version control, and
three RepoRadar run files under `evals/results/`, which are not -- the same asymmetry that
`gold_targets.json` and `multisource_arm.json` exist to close.

The artifact deliberately records more than the headline. The headline (+1.08 paired over 37)
is the least informative number in the file: it is a blend of a cohort where RepoRadar wins,
a cohort where it loses, and a split -- over-answered vs not -- that accounts for all of it.
"""

import json
import pathlib
import random
import statistics as st
import sys

sys.path.insert(0, "evals")
from reporadar.paper_id import is_arxiv_id

RES = pathlib.Path("evals/results")
ARMS = {
    "arxiv": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json",
    "arxiv_epmc": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json",
    "arxiv_openalex": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260828T052915Z.json",
}
PRIMARY = "arxiv_epmc"  # the P24 arm; the one a shipped default would use
OPUS5 = pathlib.Path("evals/gold_spread_v2_opus5.json")
SEED = 20260828
RESAMPLES = 4000

COHORTS = {
    "core25": lambda c: not c.startswith(("bio-", "mat-")),
    "bio6": lambda c: c.startswith("bio-"),
    "matsci6": lambda c: c.startswith("mat-"),
    "scientific12": lambda c: c.startswith(("bio-", "mat-")),
    "all37": lambda _c: True,
}


def pt(score: int) -> int:
    """net@2: +1 for an actionable paper, -2 for a non-actionable one."""
    return 1 if int(score) >= 2 else -2


def net(scores) -> int:
    return sum(pt(s) for s in scores)


def ci(deltas):
    """Paired bootstrap, seeded. Same estimator as every other arm in this project."""
    rnd = random.Random(SEED)
    b = sorted(st.mean(rnd.choices(deltas, k=len(deltas))) for _ in range(RESAMPLES))
    return round(b[int(0.025 * RESAMPLES)], 2), round(b[int(0.975 * RESAMPLES) - 1], 2)


# -- RepoRadar: three source arms, same day, same flags but `sources` --
arms = {}
for label, fname in ARMS.items():
    run = json.loads((RES / fname).read_text(encoding="utf-8"))
    arms[label] = {
        "run_file": fname,
        "sources": run[0]["sources"],
        "digest_window": run[0]["digest_window"],
        "w_embedding": run[0]["w_embedding"],
        "per_case": {
            e["case"]: [
                (str(p["arxiv_id"]), int(p["judge_score"]))
                for p in e["returned"]["reporadar_toppicks"]
            ]
            for e in run
        },
    }

# -- Opus 5: draw 1 of the v2 / 30-turn sweep --
rows = json.loads(OPUS5.read_text(encoding="utf-8"))["results"]
o5_picks, o5_cost = {}, {}
for key, row in rows.items():
    draw, case = key.split("/", 1)
    if draw != "1" or row.get("status") != "ok":
        continue
    scores = row.get("scores") or {}
    # Model output order, restricted to picks that actually got a verdict. Hallucinated and
    # unjudgeable picks are ABSENT from net@2 rather than scored negative -- the void-not-null
    # rule. On the materials six that costs nothing: zero of either.
    o5_picks[case] = [(p, scores[p]) for p in (row.get("picks") or []) if p in scores]
    o5_cost[case] = row.get("cost_usd")

cases = sorted(set(o5_picks) & set(arms[PRIMARY]["per_case"]))


def block(picks_by_case, sel):
    """Volume, precision and provenance for one system over one cohort.

    `arxiv_precision` is separated from `precision` because the question P26 was asked --
    is Opus 5 winning through non-arXiv material? -- cannot be answered by the pooled figure.
    """
    ps = [p for c in sel for p in picks_by_case[c]]
    na = [(p, s) for p, s in ps if not is_arxiv_id(p)]
    ax = [(p, s) for p, s in ps if is_arxiv_id(p)]
    return {
        "mean_net2": round(st.mean(net(s for _, s in picks_by_case[c]) for c in sel), 2),
        "shown": len(ps),
        "shown_per_case": round(len(ps) / len(sel), 1),
        "precision": round(sum(1 for _, s in ps if s >= 2) / len(ps), 3) if ps else None,
        "arxiv_precision": round(sum(1 for _, s in ax if s >= 2) / len(ax), 3) if ax else None,
        "n_non_arxiv": len(na),
        "non_arxiv_share": round(len(na) / len(ps), 3) if ps else None,
        "non_arxiv_precision": (
            round(sum(1 for _, s in na if s >= 2) / len(na), 3) if na else None
        ),
        "non_arxiv_net2_per_case": round(net(s for _, s in na) / len(sel), 2),
    }


out = {
    "_comment": (
        "P26: Opus 5 as a comparator over the COMPLETE 37-case cohort, against RepoRadar at "
        "three source configurations. Derived by evals/freeze_opus5_arm.py from "
        "evals/gold_spread_v2_opus5.json (in tree) and three run files under evals/results/ "
        "(gitignored); pinned by tests/test_opus5_arm.py. net@2 = #actionable - 2 x "
        "#non-actionable over what each system RETURNED; abstaining scores 0."
    ),
    "opus5_config": {
        "artifact": OPUS5.name,
        "draw": 1,
        "prompt_version": "v2",
        "max_turns": 30,
        "model": "claude-opus-5",
        "effort": None,
        "cli_auth": "subscription",
        "n_cases": len(cases),
        "cost_usd": round(sum(v for v in o5_cost.values() if v), 2),
    },
    "reporadar_arms": {
        label: {k: a[k] for k in ("run_file", "sources", "digest_window", "w_embedding")}
        for label, a in arms.items()
    },
    "per_case": {
        c: {
            "opus5": net(s for _, s in o5_picks[c]),
            "opus5_n": len(o5_picks[c]),
            **{label: net(s for _, s in arms[label]["per_case"][c]) for label in ARMS},
            **{f"{label}_n": len(arms[label]["per_case"][c]) for label in ARMS},
        }
        for c in cases
    },
    "cohorts": {},
}

for name, pred in COHORTS.items():
    sel = [c for c in cases if pred(c)]
    d = [
        net(s for _, s in arms[PRIMARY]["per_case"][c]) - net(s for _, s in o5_picks[c])
        for c in sel
    ]
    lo, hi = ci(d)
    out["cohorts"][name] = {
        "n_cases": len(sel),
        "opus5": block(o5_picks, sel),
        **{label: block(arms[label]["per_case"], sel) for label in ARMS},
        "paired_delta_primary_minus_opus5": round(st.mean(d), 2),
        "ci95": [lo, hi],
        "wins": sum(1 for x in d if x > 0),
        "losses": sum(1 for x in d if x < 0),
    }

# -- where the margin lives --
# The headline is a blend. These two splits are the finding: on the cases where Opus 5 does
# not over-answer the two systems are level, and every point of the margin comes from cases
# where it does -- four of which are cases RepoRadar answers by abstaining entirely.
rr = {c: net(s for _, s in arms[PRIMARY]["per_case"][c]) for c in cases}
o5 = {c: net(s for _, s in o5_picks[c]) for c in cases}
total = sum(rr[c] - o5[c] for c in cases)
splits = {
    "opus5_overanswered": [c for c in cases if o5[c] < 0],
    "opus5_not_overanswered": [c for c in cases if o5[c] >= 0],
    "reporadar_abstained": [c for c in cases if not arms[PRIMARY]["per_case"][c]],
    "reporadar_answered": [c for c in cases if arms[PRIMARY]["per_case"][c]],
}
out["margin_decomposition"] = {}
for name, sel in splits.items():
    d = [rr[c] - o5[c] for c in sel]
    out["margin_decomposition"][name] = {
        "cases": sorted(sel),
        "n_cases": len(sel),
        "reporadar_mean": round(st.mean(rr[c] for c in sel), 2) if sel else None,
        "opus5_mean": round(st.mean(o5[c] for c in sel), 2) if sel else None,
        "paired_delta": round(st.mean(d), 2) if sel else None,
        "share_of_total_margin": round(sum(d) / total, 3) if sel and total else None,
    }

pathlib.Path("evals/opus5_arm.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            "cohorts": {
                k: {
                    kk: v[kk]
                    for kk in (
                        "n_cases",
                        "paired_delta_primary_minus_opus5",
                        "ci95",
                        "wins",
                        "losses",
                    )
                }
                for k, v in out["cohorts"].items()
            },
            "margin_decomposition": out["margin_decomposition"],
        },
        indent=1,
    )
)
