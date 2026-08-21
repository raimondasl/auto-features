"""§25: why does gate-score 3 mean something different on scientific software? ($0 primary.)

    uv run python evals/score3_mechanism.py             # primary + secondary + tertiary, $0
    uv run python evals/score3_mechanism.py --validity  # + Sonnet on unlabelled score-3, ~$0.40

§16.5 measured that gate-score-3 papers are non-actionable 31% of the time on scientific
software against 7% on ML/CS (Fisher p = 0.027), and §17.3 showed that survives a judge swap.
§18.5 found the judge-free companion: the gate *emits* score 3 at 20.0% against 8.0%. So the
asymmetry is settled. This asks **why**, because §9.4 killed both global repairs and warned that
with nine misses, any variant that works after three attempts is fitting the set. A mechanism
tells you which repair to build; a fourth variant tells you nothing.

**The hypothesis, and the fact that already complicates it.** §0 and §6's G1 have asserted since
§5 that the gate reads a tool-name match as relevance — "six *use* CHGNet/MACE and name it in the
abstract ... the gate scores the name-match 3, the judge scores it 1". Probed before §25's bars
were written, predictor only: scientific score-3 papers name the tool 69% of the time and ML/CS
ones 60%. **Both domains name it at similar rates**, so a main effect cannot by itself explain a
31%-versus-7% split — the story needs an interaction, over cells too small to test. That is why
the tertiary below carries no bar and why a NULL on the primary is worth as much as a WIN: it
would refute a mechanism this document has carried for twenty sections.

Everything here except `--validity` reads labels already bought and abstracts already on disk.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from harness import load_benchmark  # noqa: E402
from label_pool import fisher_exact  # noqa: E402
from second_judge import (  # noqa: E402
    ACTIONABLE,
    CACHE,
    DEFAULT_MODEL,
    second_verdict,
    verify_contexts,
)
from second_judge import _load_env as load_env  # noqa: E402

from reporadar.llm_client import LLMError  # noqa: E402

RESULTS = EVALS / "results"
# The five runs §25.3 names. `pool` is where each one's abstracts live.
RUNS = (
    (
        "cohort3-sci",
        "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T060917Z.json",
        "pool-cohort3",
    ),
    (
        "cohort3-legacy",
        "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T172033Z.json",
        "pool-cohort3",
    ),
    (
        "epmc-control",
        "judge-gpt-5.5-frozenpool-bigrams_verified-20260821T045009Z.json",
        "pool-epmc-control",
    ),
    (
        "epmc-treat",
        "judge-gpt-5.5-frozenpool-bigrams_verified-20260821T052215Z.json",
        "pool-epmc-treat",
    ),
    (
        "window30",
        "judge-gpt-5.5-frozenpool-bigrams_verified-20260821T152858Z.json",
        "pool-epmc-treat",
    ),
)
GATE_TOP = 3
# §25.3: the one tool name that is also an ordinary English word. The primary is reported with
# and without it, and §25.5 kills the predictor if the two differ by more than this.
AMBIGUOUS = {"mace"}
KILL_SHIFT = 0.10
WIN_GAP = 0.20  # §25.5


def scientific(case: str) -> bool:
    return case.startswith(("bio-", "mat-"))


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def collect() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every unique (case, paper) the gate scored 3 in any run, with its abstract and label.

    A paper is included if it was scored 3 in ANY run, which is what §25.3 declared. The gate is
    sampled, so the same paper can score 3 in one run and 2 in another; those conflicts are
    counted and reported rather than resolved silently.
    """
    bench = {c["name"]: c for c in load_benchmark()["cases"]}
    pools: dict[tuple[str, str], dict[str, Any]] = {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    seen_any: dict[tuple[str, str], set[int]] = {}
    stats = Counter()

    for _tag, fname, pool_dir in RUNS:
        path = RESULTS / fname
        if not path.is_file():
            stats["missing_run"] += 1
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            case = rec["case"]
            key_pool = (pool_dir, case)
            if key_pool not in pools:
                p = EVALS / ".work" / pool_dir / f"{case}.json"
                pools[key_pool] = (
                    {
                        x["arxiv_id"]: x
                        for x in json.loads(p.read_text(encoding="utf-8"))["candidates"]
                    }
                    if p.is_file()
                    else {}
                )
            tool = bench[case]["live_repo"].rstrip("/").split("/")[-1]
            for x in rec["returned"]["reporadar_top10"]:
                gate = x.get("llm_score")
                if gate is None:
                    continue
                k = (case, x["arxiv_id"])
                seen_any.setdefault(k, set()).add(gate)
                if gate != GATE_TOP or k in rows:
                    continue
                abstract = pools[key_pool].get(x["arxiv_id"], {}).get("abstract", "")
                if not abstract:
                    stats["no_abstract"] += 1
                    continue
                names = bool(
                    re.search(rf"\b{re.escape(tool)}\b", f"{x.get('title', '')} {abstract}", re.I)
                )
                rows[k] = {
                    "case": case,
                    "arxiv_id": x["arxiv_id"],
                    "title": x.get("title", ""),
                    "tool": tool,
                    "names_tool": names,
                    "ambiguous": tool.lower() in AMBIGUOUS,
                    "judge_score": x["judge_score"],
                    "non_actionable": x["judge_score"] < ACTIONABLE,
                    "scientific": scientific(case),
                    "pool_dir": pool_dir,
                }
    stats["gate_conflicts"] = sum(
        1 for k, v in seen_any.items() if GATE_TOP in v and len(v) > 1 and k in rows
    )
    return list(rows.values()), dict(stats)


def compare(
    label: str, a: list[dict[str, Any]], b: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    """Two-proportion comparison on *field*, with Wilson intervals and an exact p."""
    ka, na = sum(1 for r in a if r[field]), len(a)
    kb, nb = sum(1 for r in b if r[field]), len(b)
    if not na or not nb:
        print(f"  {label:34} one arm empty ({na} vs {nb}) — not reported")
        return {}
    la, ha = wilson(ka, na)
    lb, hb = wilson(kb, nb)
    gap = ka / na - kb / nb
    p = fisher_exact(ka, na - ka, kb, nb - kb)
    print(
        f"  {label:34} {ka:3d}/{na:3d} = {ka / na:.3f} [{la:.3f},{ha:.3f}]   vs   "
        f"{kb:3d}/{nb:3d} = {kb / nb:.3f} [{lb:.3f},{hb:.3f}]   gap {gap:+.3f}  p={p:.4f}"
    )
    return {"gap": gap, "p": p, "a": [ka, na], "b": [kb, nb]}


def emission(fname: str, pred: Any = None) -> Counter:
    """Gate score histogram over a run's ranked window. No labels are read."""
    c: Counter = Counter()
    for rec in json.loads((RESULTS / fname).read_text(encoding="utf-8")):
        if pred and not pred(rec["case"]):
            continue
        for x in rec["returned"]["reporadar_top10"]:
            c[x.get("llm_score")] += 1
    return c


def judge_free_companion() -> None:
    """§25.4's judge-free endpoint: does §18.5's emission asymmetry hold on newer runs?

    Two things this decomposition establishes and §18.5 could not, because §18.5 pooled the
    twelve scientific cases: the asymmetry is carried by MATSCI, and "the gate never emits 0 on
    scientific software" was a property of the top-15 WINDOW rather than of the domain.
    """
    leg = emission(RUNS[1][1])
    nl = sum(leg.values())
    print("\n" + "=" * 100)
    print("JUDGE-FREE COMPANION — gate emission of score 3. No labels are involved.")
    print("=" * 100)
    print(f"  {'legacy-25 (baseline)':30} {leg[3]:3d}/{nl} = {leg[3] / nl:5.1%}")
    arms = (
        ("cohort3 MATSCI only", RUNS[0][1], lambda c: c.startswith("mat-")),
        ("cohort3 BIO only", RUNS[0][1], lambda c: c.startswith("bio-")),
        ("epmc-control (bio, fresh)", RUNS[2][1], None),
        ("epmc-treat (bio + epmc)", RUNS[3][1], None),
    )
    for label, fname, pred in arms:
        c = emission(fname, pred)
        n = sum(c.values())
        p = fisher_exact(c[3], n - c[3], leg[3], nl - leg[3])
        print(f"  {label:30} {c[3]:3d}/{n:3d} = {c[3] / n:5.1%}   vs legacy p={p:.4f}")
    deep = emission(RUNS[4][1])
    print(
        f"\n  §18.5 also reported 'the gate never emits 0 on scientific software' (0/180). At a\n"
        f"  30-deep window the same six repositories emit {deep[0]} zeros, so that was a\n"
        "  property of the top-15 window and not of the domain. A qualification of §18.5."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--validity", action="store_true", help="second-judge unlabelled score-3 (~$0.40)"
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows, stats = collect()
    sci = [r for r in rows if r["scientific"]]
    leg = [r for r in rows if not r["scientific"]]
    print(
        f"population: {len(rows)} unique gate-score-3 papers   "
        f"scientific {len(sci)}   ML/CS {len(leg)}"
    )
    print(f"  dropped for no abstract: {stats.get('no_abstract', 0)}")
    print(
        f"  scored 3 in one run and something else in another: {stats.get('gate_conflicts', 0)}"
        "  (the gate is sampled; included per §25.3)"
    )

    print("\n" + "=" * 100)
    print("PRIMARY — does naming the tool predict NON-ACTIONABILITY among score-3 papers?")
    print("=" * 100)
    named = [r for r in rows if r["names_tool"]]
    unnamed = [r for r in rows if not r["names_tool"]]
    full = compare("all 85: names tool vs not", named, unnamed, "non_actionable")

    clean = [r for r in rows if not r["ambiguous"]]
    sub = compare(
        "excluding `mace`",
        [r for r in clean if r["names_tool"]],
        [r for r in clean if not r["names_tool"]],
        "non_actionable",
    )
    shift = abs(full.get("gap", 0.0) - sub.get("gap", 0.0))
    print(f"\n  KILL check: excluding `mace` moves the gap by {shift:.3f}  (bar {KILL_SHIFT:.2f})")
    if shift > KILL_SHIFT:
        print("    KILL — the predictor is measuring string luck; no conclusion may be drawn.")
        return 1
    print("    PASS — the predictor is not carried by one ambiguous name.")
    verdict = "WIN" if full.get("gap", 0.0) >= WIN_GAP else "NULL"
    print(f"\n  PRIMARY VERDICT: {verdict}   (WIN bar: gap >= {WIN_GAP:+.2f})")
    if verdict == "NULL":
        print(
            "    §0's and §6 G1's mechanism is UNSUPPORTED on this evidence. That is a\n"
            "    correction to a belief this document has carried since §5, and §25.5 said in\n"
            "    advance it would be worth as much as a positive result."
        )

    print("\n" + "=" * 100)
    print("SECONDARY — does the domain asymmetry replicate?")
    print("=" * 100)
    compare("all: scientific vs ML/CS", sci, leg, "non_actionable")
    arxiv_only = [r for r in rows if r["pool_dir"] == "pool-cohort3"]
    compare(
        "arXiv-only (§16.5's population)",
        [r for r in arxiv_only if r["scientific"]],
        [r for r in arxiv_only if not r["scientific"]],
        "non_actionable",
    )

    print("\n" + "=" * 100)
    print("TERTIARY — the interaction. NO BAR: §25.2 declared these cells too small.")
    print("=" * 100)
    for dom, label in ((True, "scientific"), (False, "ML/CS")):
        d = [r for r in rows if r["scientific"] is dom]
        compare(
            f"{label}: names tool vs not",
            [r for r in d if r["names_tool"]],
            [r for r in d if not r["names_tool"]],
            "non_actionable",
        )
    print("  Reported as magnitudes. No verdict is available at these cell sizes.")

    judge_free_companion()

    if args.validity:
        todo = [
            r
            for r in rows
            if not (
                CACHE / args.model / r["case"] / f"{r['arxiv_id'].replace('/', '_')}.json"
            ).is_file()
        ]
        print(f"\nVALIDITY — second-judging {len(todo)} score-3 papers with no Sonnet label")
        if todo:
            load_env()
            contexts, drifted = verify_contexts(sorted({r["case"] for r in todo}))
            todo = [r for r in todo if r["case"] not in set(drifted)]
            pools: dict[tuple[str, str], dict[str, Any]] = {}
            for r in todo:
                kp = (r["pool_dir"], r["case"])
                if kp not in pools:
                    p = EVALS / ".work" / r["pool_dir"] / f"{r['case']}.json"
                    pools[kp] = {
                        x["arxiv_id"]: x
                        for x in json.loads(p.read_text(encoding="utf-8"))["candidates"]
                    }
                r["paper"] = pools[kp].get(r["arxiv_id"])
            todo = [r for r in todo if r.get("paper")]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(
                        second_verdict, r["case"], contexts[r["case"]], r["paper"], args.model
                    ): r
                    for r in todo
                }
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except (LLMError, ValueError, KeyError) as exc:
                        print(f"  ! {futs[fut]['case']}: {str(exc)[:80]}")
        for r in rows:
            p = CACHE / args.model / r["case"] / f"{r['arxiv_id'].replace('/', '_')}.json"
            if p.is_file():
                s = int(json.loads(p.read_text(encoding="utf-8"))["score"])
                r["sonnet_non_actionable"] = s < ACTIONABLE
        have = [r for r in rows if "sonnet_non_actionable" in r]
        print(f"  {len(have)}/{len(rows)} score-3 papers carry a Sonnet label")
        print("\n  PRIMARY under Sonnet:")
        compare(
            "names tool vs not",
            [r for r in have if r["names_tool"]],
            [r for r in have if not r["names_tool"]],
            "sonnet_non_actionable",
        )

    out = EVALS / ".work" / "score3_mechanism.json"
    out.write_text(
        json.dumps(
            {"n": len(rows), "rows": [{k: v for k, v in r.items() if k != "paper"} for r in rows]},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
