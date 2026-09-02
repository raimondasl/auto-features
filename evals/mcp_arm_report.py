"""Score the augmented arm against RepoRadar alone and against Opus 5 alone. [P27]

Four columns over one cohort, one judge, one metric:

| | what it is |
|---|---|
| **A** | RepoRadar's frozen arXiv+EPMC arm — the published number |
| **A′** | the same run as the **product** would display it: minus papers the repo already cites |
| **B** | Opus 5, v2 prompt, 30 turns, WebSearch+WebFetch — `gold_spread_v2_opus5.json` draw 1 |
| **C** | B with RepoRadar's MCP server attached — `gold_spread_v2_opus5_web_rr.json` |

**A′ exists because C's tool serves A′, not A.** The product mutes papers the repository's
own README, CITATION file or bibliography already cites; the benchmark harness has no such
rule, so arm A is scored on picks a user would never be shown. Comparing C against A would
charge the augmented arm for papers it was never given. A is reported beside A′ because A
is what every published figure quotes — this does not restate it away (C-17), it says what
each number is.

**C is void until it is run**, never zero. An unrun arm scored as 0 net@2 would read as
"the agent recommended nothing", which is the failure this project keeps paying for; the
artifact records `status: "not_run"` and the comparison sections are absent.

    uv run python evals/mcp_arm_report.py            # $0, rewrite evals/mcp_arm.json
    uv run python evals/mcp_arm_report.py --print    # $0, and show the table
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from band_testbeds import sign_test  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402
from harness import WORK_DIR  # noqa: E402
from rr_mcp_arm import ARM_A  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.profiler import cited_arxiv_ids_of  # noqa: E402

ARM_B = EVALS / "gold_spread_v2_opus5.json"
ARM_C = EVALS / "gold_spread_v2_opus5_web_rr.json"
# C-wide: the SAME arm with the MCP store's corpus widened from the digest picks to the
# whole frozen pool, and nothing else changed. `get_ranked_papers` is byte-identical
# between the two stores (proved per case by `rr_mcp_arm.compare_stores`), so a C-wide
# minus C difference can only be `search_papers`. Registered in the PREREG addendum.
ARM_C_WIDE = EVALS / "gold_spread_v2_opus5_web_rrwide.json"
OUT = EVALS / "mcp_arm.json"

COHORTS = {
    "core25": lambda c: not c.startswith(("bio-", "mat-")),
    "bio6": lambda c: c.startswith("bio-"),
    "matsci6": lambda c: c.startswith("mat-"),
    # The cohort the arm was actually run over first, and the one where its two halves
    # disagree. Named rather than left to be inferred from `all37 PARTIAL`.
    "scientific12": lambda c: c.startswith(("bio-", "mat-")),
    "all37": lambda _c: True,
}


def pt(score: int) -> int:
    """net@2: +1 for an actionable paper, −2 for a non-actionable one."""
    return 1 if int(score) >= 2 else -2


def net(scores) -> int:
    return sum(pt(s) for s in scores)


def ci(deltas: list[float]) -> tuple[float, float]:
    lo, hi = paired_bootstrap([float(x) for x in deltas])
    return round(lo, 2), round(hi, 2)


def reporadar_arms() -> tuple[dict[str, list], dict[str, list], dict[str, int]]:
    """(A, A′, muted counts) per case, from the frozen run and the repositories themselves.

    The already-cited set is read from the clone with `cited_arxiv_ids_of` — the same
    function `notify` uses, rather than a second implementation of the rule.
    """
    rows = json.loads(ARM_A.read_text(encoding="utf-8"))
    a: dict[str, list] = {}
    a_prime: dict[str, list] = {}
    muted: dict[str, int] = {}
    for row in rows:
        case = row["case"]
        picks = [
            (str(p["arxiv_id"]), int(p["judge_score"]))
            for p in row["returned"]["reporadar_toppicks"]
            if p.get("judge_score") is not None
        ]
        repo = WORK_DIR / case
        cited = {dedup_id(c) for c in (cited_arxiv_ids_of(repo) if repo.exists() else ())}
        kept = [(pid, s) for pid, s in picks if dedup_id(pid) not in cited]
        a[case], a_prime[case] = picks, kept
        muted[case] = len(picks) - len(kept)
    return a, a_prime, muted


def agent_arm(path: Path) -> tuple[dict[str, list], dict[str, Any]]:
    """(picks per case, meta) for one `gold_spread` artifact's draw 1.

    Hallucinated and unjudgeable picks are ABSENT from net@2 rather than scored negative —
    void, not null, the same rule `freeze_opus5_arm` applies.
    """
    stored = json.loads(path.read_text(encoding="utf-8"))
    picks: dict[str, list] = {}
    cost = 0.0
    calls: dict[str, int] = {}
    n_zero_call_rows = 0
    for key, row in stored["results"].items():
        draw, case = key.split("/", 1)
        if draw != "1" or row.get("status") != "ok":
            continue
        scores = row.get("scores") or {}
        picks[case] = [(p, scores[p]) for p in (row.get("picks") or []) if p in scores]
        cost += row.get("cost_usd") or 0.0
        mcp = row.get("mcp")
        if mcp is not None:
            for tool, n in mcp["by_tool"].items():
                calls[tool] = calls.get(tool, 0) + n
            n_zero_call_rows += int(mcp["n"] == 0)
    meta: dict[str, Any] = {
        "artifact": path.name,
        "draw": 1,
        "prompt_version": stored.get("prompt_version"),
        "model": stored.get("model"),
        "tools": stored.get("tools", "web"),
        "n_cases": len(picks),
        "cost_usd": round(cost, 2),
    }
    if stored.get("tools", "web") != "web":
        # The kill condition's own evidence, on the artifact rather than in a log a reader
        # would have to go and find. More than 3 zero-call rows and the sweep is a
        # discoverability result, not a measurement of RepoRadar-plus-agent.
        meta["mcp_calls"] = dict(sorted(calls.items()))
        meta["mcp_calls_total"] = sum(calls.values())
        meta["rows_with_zero_mcp_calls"] = n_zero_call_rows
        meta["treatment_present"] = n_zero_call_rows <= 3
    return picks, meta


def provenance(picks_by_case: dict[str, list], sel: list[str]) -> dict[str, Any]:
    """Where an agent arm's picks came from: the digest, the wider pool, or off-pool.

    The registered secondary for the wide arm, and the only direct evidence of whether a
    wider `search_papers` corpus fed the agent anything it could not have had in C. A
    headline difference is an inference; this is a count.

    Three buckets, and the middle one is the whole question:

    * **digest** — the paper was in RepoRadar's Top Picks, so the agent could have got it
      from `get_ranked_papers` in EITHER arm;
    * **pool_only** — in the frozen candidate pool but not the digest, so in the wide arm it
      was reachable through `search_papers` and in the narrow arm it was not;
    * **off_pool** — the agent found it somewhere else entirely (its own web search).

    Ids are compared with `dedup_id`, the project's one normaliser (C-14). A non-arXiv pick
    cannot match an arXiv pool id and lands in `off_pool`, which is correct: the pool's
    Europe PMC entries are stored under their own ids and a DOI-shaped pick is by
    construction something the arXiv-shaped pool did not offer under that name.
    """
    from rr_mcp_arm import _pool_path, arm_picks

    counts = {"digest": 0, "pool_only": 0, "off_pool": 0}
    by_case: dict[str, dict[str, int]] = {}
    for case in sel:
        digest = {dedup_id(p["arxiv_id"]) for p in arm_picks(ARM_A, case)}
        pool_json = json.loads(_pool_path(ARM_A, case).read_text(encoding="utf-8"))
        pool = {dedup_id(c["arxiv_id"]) for c in pool_json["candidates"]}
        here = {"digest": 0, "pool_only": 0, "off_pool": 0}
        for pid, _score in picks_by_case[case]:
            norm = dedup_id(pid)
            bucket = "digest" if norm in digest else ("pool_only" if norm in pool else "off_pool")
            here[bucket] += 1
            counts[bucket] += 1
        by_case[case] = here
    total = sum(counts.values())
    return {
        **counts,
        "n_picks": total,
        "digest_share": round(counts["digest"] / total, 3) if total else None,
        "pool_only_share": round(counts["pool_only"] / total, 3) if total else None,
        "by_case": by_case,
    }


def block(picks_by_case: dict[str, list], sel: list[str]) -> dict[str, Any]:
    ps = [p for c in sel for p in picks_by_case[c]]
    return {
        "mean_net2": round(st.mean(net(s for _, s in picks_by_case[c]) for c in sel), 2),
        "shown": len(ps),
        "shown_per_case": round(len(ps) / len(sel), 1),
        "precision": round(sum(1 for _, s in ps if s >= 2) / len(ps), 3) if ps else None,
        "abstained_on": sum(1 for c in sel if not picks_by_case[c]),
    }


def paired(x: dict[str, list], y: dict[str, list], sel: list[str]) -> dict[str, Any]:
    """x − y, per case, with the project's own estimators rather than new ones (C-25)."""
    d = [float(net(s for _, s in x[c]) - net(s for _, s in y[c])) for c in sel]
    lo, hi = ci(d)
    return {
        "mean": round(st.mean(d), 2),
        "ci95": [lo, hi],
        "excludes_zero": lo > 0 or hi < 0,
        "wins": sum(1 for v in d if v > 0),
        "losses": sum(1 for v in d if v < 0),
        "ties": sum(1 for v in d if v == 0),
        # Both estimators the pre-registration names, not one of them. They answer
        # different questions -- the interval is about magnitude, the sign test about
        # direction -- and at n = 6 the second is the one that can say almost nothing.
        "sign_test_p": round(sign_test(d)["p"], 4),
        "per_case": {c: v for c, v in zip(sel, d, strict=True)},
    }


def build() -> dict[str, Any]:
    a, a_prime, muted = reporadar_arms()
    b, b_meta = agent_arm(ARM_B)

    out: dict[str, Any] = {
        "_comment": (
            "P27: RepoRadar alone (A), RepoRadar as the product displays it (A'), Opus 5 "
            "alone (B), and Opus 5 with RepoRadar's MCP server attached (C). Derived by "
            "evals/mcp_arm_report.py; pinned by tests/test_mcp_arm_report.py. net@2 = "
            "#actionable - 2 x #non-actionable over what each system RETURNED; abstaining "
            "scores 0. Pre-registration: evals/PREREG-mcp-arm.md."
        ),
        "arms": {
            "A": {"source": ARM_A.name, "what": "RepoRadar, frozen arXiv+EPMC arm"},
            "A_prime": {
                "source": ARM_A.name,
                "what": "A minus picks the repository already cites — what the product shows",
                "n_muted": sum(muted.values()),
                "n_picks": sum(len(v) for v in a.values()),
                "muted_by_case": {c: n for c, n in sorted(muted.items()) if n},
            },
            "B": b_meta,
        },
        "cohorts": {},
    }

    # The augmented arms, keyed by column name. A list rather than two special cases,
    # because a third one is exactly how the second would have been bolted on wrongly:
    # `agent_arm` is shared, the partial-cohort discipline below is shared, and an arm
    # that has not run must produce `not_run` rather than a column of zeros in both.
    agent_arms: dict[str, dict[str, list]] = {}
    for label, path, seed_flags, tools in (
        ("C", ARM_C, "--seed", "web+rr"),
        ("C_wide", ARM_C_WIDE, "--seed --wide", "web+rrwide"),
    ):
        if path.exists():
            picks, meta = agent_arm(path)
            agent_arms[label] = picks
            out["arms"][label] = meta
        else:
            # Void, not zero. An unrun arm reported as 0 net@2 reads as "the agent
            # recommended nothing", which is a measurement, and this is its absence.
            out["arms"][label] = {
                "status": "not_run",
                "artifact": path.name,
                "how": (
                    f"uv run python evals/rr_mcp_arm.py {seed_flags}  (free), then "
                    f"uv run python evals/gold_spread.py --tools {tools} --prompt-version "
                    "v2 --max-turns 30 --baseline-model claude-opus-5 --cohort all --draws 1"
                ),
            }
    c = agent_arms.get("C")
    c_wide = agent_arms.get("C_wide")

    # A n B, NOT A n B n C. Intersecting with a partially-run arm C would silently
    # shrink every other column to whatever C happens to have finished -- so a 6-case
    # matsci run relabels itself `all37` and reports matsci's levels under that name.
    # The arm is meant to be run cohort by cohort as quota allows, which makes a
    # partial C the default state rather than an edge case.
    shared = sorted(set(a) & set(b))
    for name, pred in COHORTS.items():
        sel = [x for x in shared if pred(x)]
        if not sel:
            continue
        entry: dict[str, Any] = {
            "n_cases": len(sel),
            "A": block(a, sel),
            "A_prime": block(a_prime, sel),
            "B": block(b, sel),
            "A_minus_B": paired(a, b, sel),
            "A_prime_minus_B": paired(a_prime, b, sel),
            # What the already-cited mute costs arm A. Reported as its own line because it
            # is the difference between the published comparison and the shipped one.
            "A_minus_A_prime": paired(a, a_prime, sel),
        }
        sel_c = [x for x in sel if c is not None and x in c]
        if sel_c:
            # C's own cohort, named. Every C figure is over `n_cases_c`, which is NOT
            # `n_cases` until the sweep is complete -- and a partial cohort says so,
            # rather than leaving a reader to compare a 6-case mean against a 37-case one.
            entry["n_cases_c"] = len(sel_c)
            entry["c_complete"] = len(sel_c) == len(sel)
            entry["cases_c"] = sel_c
            entry["C"] = block(c, sel_c)
            entry["C_minus_B"] = paired(c, b, sel_c)  # the registered primary
            entry["C_minus_A_prime"] = paired(c, a_prime, sel_c)
            # A' and B restricted to exactly C's cases, so the three-way comparison is
            # over ONE case set. Without them a reader compares C against a B computed
            # on a different cohort, which is what the old intersection was doing.
            entry["B_on_c_cases"] = block(b, sel_c)
            entry["A_prime_on_c_cases"] = block(a_prime, sel_c)
            entry["C_provenance"] = provenance(c, sel_c)
            entry["B_provenance"] = provenance(b, sel_c)
        sel_w = [x for x in sel if c_wide is not None and x in c_wide]
        if sel_w:
            entry["n_cases_c_wide"] = len(sel_w)
            entry["c_wide_complete"] = len(sel_w) == len(sel)
            entry["C_wide"] = block(c_wide, sel_w)
            entry["C_wide_minus_B"] = paired(c_wide, b, sel_w)
            entry["C_wide_provenance"] = provenance(c_wide, sel_w)
            # THE registered primary for the wide arm, and the only one that isolates the
            # corpus: everything else about C and C_wide is identical, `get_ranked_papers`
            # included. Restricted to the cases BOTH arms ran, because a paired statistic
            # over a case only one of them covered is not paired.
            both = [x for x in sel_w if c is not None and x in c]
            if both:
                entry["n_cases_both"] = len(both)
                entry["C_wide_minus_C"] = paired(c_wide, c, both)
                entry["C_on_both"] = block(c, both)
                entry["C_wide_on_both"] = block(c_wide, both)
        out["cohorts"][name] = entry
    return out


def show(art: dict[str, Any]) -> None:
    any_c = any("C" in e for e in art["cohorts"].values())
    any_w = any("C_wide" in e for e in art["cohorts"].values())
    head = f"{'cohort':<12} {'n':>3}  {'A':>6} {chr(39) + 'A':>6} {'B':>6}"
    if any_c:
        head += f"  {'nC':>3} {'C':>6} {'B|C':>6}  {'C-B':>7}  {'95% CI':>16}"
    if any_w:
        head += f"  {'Cwide':>6}  {'Cw-C':>6}  {'95% CI':>16}"
    print(head)
    for name, e in art["cohorts"].items():
        line = (
            f"{name:<12} {e['n_cases']:>3}  {e['A']['mean_net2']:>+6.2f} "
            f"{e['A_prime']['mean_net2']:>+6.2f} {e['B']['mean_net2']:>+6.2f}"
        )
        if "C" in e:
            d = e["C_minus_B"]
            # `B|C` is B over exactly C's cases -- on a partial sweep, the only B a C
            # figure may be compared against.
            line += (
                f"  {e['n_cases_c']:>3} {e['C']['mean_net2']:>+6.2f} "
                f"{e['B_on_c_cases']['mean_net2']:>+6.2f}  {d['mean']:>+7.2f}  "
                f"[{d['ci95'][0]:>+6.2f},{d['ci95'][1]:>+6.2f}]"
                + ("" if e["c_complete"] else "  PARTIAL")
            )
        elif any_c:
            line += f"  {'-':>3} {'-':>6} {'-':>6}  {'-':>7}  {'C not run':>16}"
        if "C_wide" in e and "C_wide_minus_C" in e:
            w = e["C_wide_minus_C"]
            line += (
                f"  {e['C_wide']['mean_net2']:>+6.2f}  {w['mean']:>+6.2f}  "
                f"[{w['ci95'][0]:>+6.2f},{w['ci95'][1]:>+6.2f}]"
            )
        elif any_w:
            line += f"  {'-':>6}  {'-':>6}  {'C-wide not run':>16}"
        print(line)
    ap = art["arms"]["A_prime"]
    print(
        f"\nthe product mutes {ap['n_muted']} of {ap['n_picks']} arm-A picks "
        f"(already cited by the repository)"
    )
    for label in ("C", "C_wide"):
        arm = art["arms"].get(label, {})
        if arm.get("status") == "not_run":
            print(f"{label}: not_run - {arm['artifact']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the three arms. $0.")
    ap.add_argument("--print", action="store_true", help="also print the table")
    args = ap.parse_args()
    art = build()
    OUT.write_text(json.dumps(art, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}")
    if args.print:
        show(art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
