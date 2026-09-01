"""Give the agentic baseline RepoRadar as an MCP tool, and score the three arms. [P27]

The comparator question in PLANS 3b is *either/or*: RepoRadar's +6.27 against Opus 5's
+5.19, paired +1.08 with an interval crossing zero. Nobody has measured the arm a user
would actually run, which is **both** — Opus 5 in agentic mode with RepoRadar's MCP server
attached, free to use its own search *and* the ranked list.

It is worth measuring because the two systems fail differently, and the split is already
known (P26): on the 32 cases where Opus 5 does not over-answer the two are level (−0.06);
all of RepoRadar's margin comes from 5 cases where Opus 5 answers into a repository it
should have abstained on. RepoRadar's edge is **abstention discipline, not discovery**. So
the augmented arm has a mechanism available to it that neither arm has alone: a gate-passed
shortlist to check its own picks against, on exactly the repositories where its picks are
what hurt it.

Three arms, one judge, one cohort:

| arm | what it is | where it comes from |
|---|---|---|
| **A. RepoRadar** | the frozen arXiv+EPMC arm | `evals/results/…234024Z.json` |
| **B. Opus 5** | v2, 30 turns, WebSearch+WebFetch | `evals/gold_spread_v2_opus5.json` |
| **C. Opus 5 + RepoRadar** | B, plus `rr mcp` over stdio | this module + `--tools web+rr` |

## What makes C a controlled arm rather than a new system

**One variable.** C is B with the MCP server attached and nothing else: byte-identical
prompt (v2), same turn cap (30), same model, same auth, same allowed web tools. The tool
list is the only difference, and the prompt does not mention it — Claude Code advertises
MCP tools in its own system prompt, so an agent that does not use them is telling us
something about discoverability, which is a finding rather than a confound. Tool-call
counts are recorded per row so "did it even use it?" is answered from the artifact.

**Arm C sees exactly what the shipped product would show for arm A's run.** The store
holds the papers arm A returned, in arm A's order, with arm A's gate scores — not a fresh
RepoRadar run. So a difference between A and C cannot be a different draw of RepoRadar
(NR-54 measured that at sd 1.44/case), and C's ceiling is legible: it is A's picks plus
whatever the agent finds.

**With one subtraction, measured rather than assumed.** The product mutes papers the
repository's own README, CITATION file or bibliography already cites; the benchmark
harness has no such rule, so arm A is scored on picks the product would not display. That
is 11 of 325 picks over the 37 cases (3.4%) across 8 cases — found by pointing Opus 5 at
this arm's own MCP server on 2026-09-01 and reading what came back, not by reasoning about
it. It is worth **+0.05 net@2/case to arm A**, CI [−0.14, +0.24]: 8 of the 11 are judged
actionable and 3 are not, so the `+1`s and the `−2`s nearly cancel. Estimated at +0.22
first, by counting only the actionable side; `evals/mcp_arm_report.py` is what corrected
it. The arm does not paper over it in either direction: the tool behaves exactly as the
product does, and the comparison names a second RepoRadar arm,

    **A′ = A minus the papers the product would mute**,

which is the set arm C's tool actually serves. A is reported beside it because A is the
published number, and a reader has to be able to see both.

**The judge's verdicts are stripped before seeding, and a test enforces it.** The frozen
arm records `judge_score` and `judge_justification` beside every pick. Seeding those would
hand the agent the answer key, and the run would look entirely normal — the single failure
mode that would invalidate the whole experiment without leaving a trace. :func:`arm_picks`
allow-lists the fields it copies rather than blocking the two it must not, because a
block-list is one new field away from leaking.

**No RepoRadar spend.** The pool is frozen, the gate verdicts are frozen, and the ranking
is deterministic given the pool — so seeding costs $0 and re-running it reproduces the same
store byte for byte.

## The one number that is re-derived rather than replayed

`paper_scores.score_total` is NOT NULL and the frozen arm does not record it (its
`STAGE_FIELDS` are `llm_score`, `finescale`, `finescale_p`). Rather than invent a value the
agent would then see, this re-runs the **real ranker** over the **frozen pool** under the
arm's own `ranking_config` — free, deterministic, and verified: :func:`seed_case` refuses
to write a store unless every pick was found in the re-ranked list, because a pick the
re-rank cannot locate means the pool or the flags moved and the store would be a plausible
artifact answering a different question.

**Known and stated:** within one `llm_score` group the order may differ from the digest's.
The eval harness ranks pool dicts that carry no `score_total`, so
`rerank_by_actionability`'s tiebreak is inert there and ties keep RRF order; here the
tiebreak is live. The *set* and the gate scores are identical; only the intra-group order
can move. `llm_reason` is void, not empty — the frozen arm never recorded the gate's prose.

    uv run python evals/rr_mcp_arm.py --seed              # $0, build every case's store
    uv run python evals/rr_mcp_arm.py --seed --case rag   # one case
    uv run python evals/rr_mcp_arm.py --verify            # $0, check what is on disk
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from harness import WORK_DIR, load_benchmark  # noqa: E402
from run_judge_eval import rank_candidates  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.store import PaperStore  # noqa: E402

# Arm A: the frozen RepoRadar run the P26 comparison uses as PRIMARY. Named here rather
# than passed in, so the three arms cannot silently come from different runs -- the
# comparison is only meaningful if C's tool serves the picks A is scored on.
ARM_A = (
    EVALS / "results" / "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json"
)
# Not a constant: every frozen row records the pool directory it was collected against,
# and reading it back is what makes the store provably built from the arm's OWN pool
# rather than from whichever directory happened to have a file of the right name.
DEFAULT_POOL_DIR = WORK_DIR / "pool-core25-epmc"
MCP_DIR = WORK_DIR / "mcp-arm"

# The fields an agent may see. An ALLOW-list: `judge_score` and `judge_justification` sit
# beside these in every frozen row, and a block-list would leak the moment the results
# schema gained a third verdict field. Nothing here is the judge's opinion.
PICK_FIELDS = ("arxiv_id", "title", "llm_score", "finescale", "finescale_p")

# Never seeded, under any circumstance. Listed by name so the test that enforces the rule
# reads as a statement about the experiment rather than a string literal in an assertion.
FORBIDDEN_FIELDS = ("judge_score", "judge_justification")


def arm_picks(results_path: Path, case: str) -> list[dict[str, Any]]:
    """Arm A's returned Top Picks for *case*, in order, **with the judge stripped**.

    Raises when the case is absent rather than returning an empty list: a case arm A never
    ran is not a case where RepoRadar recommended nothing, and seeding an empty store for
    it would give arm C an unattached MCP server and call the result a measurement.
    """
    row = _arm_row(results_path, case)
    return [{f: p[f] for f in PICK_FIELDS if f in p} for p in row["returned"]["reporadar_toppicks"]]


def _arm_row(results_path: Path, case: str) -> dict[str, Any]:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    row = next((r for r in rows if r["case"] == case), None)
    if row is None:
        raise KeyError(f"{case!r} is not in {results_path.name} — arm A never ran it")
    return row


def _pool_path(results_path: Path, case: str) -> Path:
    """The frozen pool THIS arm ranked, named by the arm's own provenance record.

    A pool with the right filename in the wrong directory would re-rank cleanly and seed a
    store that serves papers arm A never saw — the plausible-artifact failure the
    fingerprint field exists to prevent, one level up.
    """
    row = _arm_row(results_path, case)
    recorded = row["pool_provenance"].get("pool_dir")
    pool_dir = Path(recorded.replace("\\", "/")) if recorded else DEFAULT_POOL_DIR
    if not pool_dir.is_absolute():
        pool_dir = EVALS.parent / pool_dir
    path = pool_dir / f"{case}.json"
    if not path.exists():
        raise FileNotFoundError(f"{case}: arm A names {path}, which is not on disk")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["fingerprint"] != row["pool_provenance"]["fingerprint"]:
        raise SystemExit(
            f"{case}: pool fingerprint {stored['fingerprint']} does not match the arm's "
            f"{row['pool_provenance']['fingerprint']} — this is not the pool arm A ranked."
        )
    return path


def case_db(case: str) -> Path:
    """Where a case's seeded store lives. Under the repo clone, exactly where `rr` looks."""
    return WORK_DIR / case / ".reporadar" / "papers.db"


def _ranking_flags(results_path: Path, case: str) -> dict[str, Any]:
    row = _arm_row(results_path, case)
    return {**row["ranking_config"], "rr_all_time": row["pool_config"]["rr_all_time"]}


def seed_case(
    case: str,
    *,
    results_path: Path = ARM_A,
    repo_dir: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Build the store `rr mcp` will serve for *case*. Returns a provenance record.

    Every number in it is real: the papers and their metadata come from the frozen pool,
    the gate scores from the frozen arm, and `score_total` from re-ranking that pool under
    the arm's own flags. Nothing is synthesised to make an ordering come out.
    """
    bench = load_benchmark()
    spec = next(c for c in bench["cases"] if c["name"] == case)
    repo_dir = repo_dir or (WORK_DIR / case)
    db_path = db_path or case_db(case)

    picks = arm_picks(results_path, case)
    pool_path = _pool_path(results_path, case)
    candidates = json.loads(pool_path.read_text(encoding="utf-8"))["candidates"]
    flags = _ranking_flags(results_path, case)

    ranked = rank_candidates(
        repo_dir,
        candidates,
        spec["expected_categories"],
        top_n=flags["rr_pool"],
        all_time=flags["rr_all_time"],
        hybrid=flags["rr_hybrid"],
        absent_category=flags["rr_absent_category"],
        w_embedding=flags["rr_w_embedding"],
    )
    scores = {p["arxiv_id"]: score for p, score in ranked}
    by_id = {p["arxiv_id"]: p for p in candidates}

    missing = [p["arxiv_id"] for p in picks if p["arxiv_id"] not in scores]
    if missing:
        # Void, not null. A pick the re-rank cannot place means the pool or the flags moved
        # under this arm, and a store written anyway would be a plausible artifact serving
        # a set that is no longer arm A's. Refuse rather than seed a partial one.
        raise SystemExit(
            f"{case}: {len(missing)} of {len(picks)} arm-A picks are absent from the "
            f"re-ranked top-{flags['rr_pool']} of {pool_path.name}: {missing[:5]}.\n"
            "The frozen pool or the ranking flags no longer reproduce the arm; refusing "
            "to seed a store that would serve a different set than arm A is scored on."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # a seed is a rebuild, never a merge into a store of unknown age
    with PaperStore(db_path) as store:
        store.upsert_papers([by_id[p["arxiv_id"]] for p in picks])
        run_id = store.record_run([f"frozen arm {results_path.name}"], len(picks), 0)
        store.save_scores(
            run_id,
            [{"arxiv_id": p["arxiv_id"], "score_total": scores[p["arxiv_id"]]} for p in picks],
        )
        store.save_llm_scores(
            run_id,
            {
                p["arxiv_id"]: {
                    "llm_score": int(p["llm_score"]),
                    # VOID, not empty. The frozen arm never recorded the gate's prose, and
                    # an empty string would read to an agent as "the gate had no reason".
                    "llm_reason": None,
                }
                for p in picks
                if p.get("llm_score") is not None
            },
        )
        finescale = {
            p["arxiv_id"]: {"finescale": p["finescale"], "finescale_p": p["finescale_p"]}
            for p in picks
            if p.get("finescale_p") is not None
        }
        if finescale:
            store.save_finescale_scores(run_id, finescale)

    return {
        "case": case,
        "db": str(db_path),
        "arm_a": results_path.name,
        "pool": pool_path.name,
        "n_picks": len(picks),
        "picks": [p["arxiv_id"] for p in picks],
        "ranking_config": flags,
    }


def write_config(case: str, *, repo_dir: Path | None = None, token: str = "") -> tuple[Path, Path]:
    """The `rr mcp` config and the MCP config `claude -p` reads. Returns (mcp_json, call_log).

    Written under `.work/mcp-arm/` rather than into the cloned repository: the agent can
    read its working tree, and a config file naming the arm would be a hint no user of the
    product would have. The store itself has to live at `<repo>/.reporadar/papers.db`
    because that is where `rr mcp` looks, and that one IS a thing a real user has.

    *token* makes the pair unique per run. The call log is how "did the agent use RepoRadar
    at all?" gets answered from the artifact instead of assumed, and a log shared between
    two concurrent runs of the same case would attribute one run's calls to the other —
    which is worse than no log, because it looks like data.
    """
    repo_dir = (repo_dir or (WORK_DIR / case)).resolve()
    MCP_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{case}-{token}" if token else case
    rr_yml = MCP_DIR / f"{stem}.reporadar.yml"
    rr_yml.write_text(
        # `triage.enabled` is what puts the gate in the payload's tiering rule. The store
        # holds only gate-passing papers, so it changes no set here -- it is set because
        # the arm must describe the configuration that produced those papers, and a reader
        # comparing this file to the frozen arm's `ranking_config` should find them agree.
        f"repo_path: {json.dumps(str(repo_dir))}\n"
        "triage:\n"
        "  enabled: true\n"
        "  min_actionable: 2\n"
        "  rerank: true\n"
        "output:\n"
        "  top_n: 15\n",
        encoding="utf-8",
    )
    call_log = MCP_DIR / f"{stem}.calls.jsonl"
    rr_bin = Path(sys.executable).with_name("rr.exe" if sys.platform == "win32" else "rr")
    mcp_json = MCP_DIR / f"{stem}.mcp.json"
    mcp_json.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "reporadar": {
                        "command": str(rr_bin),
                        "args": ["mcp", "--config", str(rr_yml)],
                        "env": {"RR_MCP_CALL_LOG": str(call_log)},
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return mcp_json, call_log


def read_call_log(path: Path) -> dict[str, Any]:
    """What the agent actually did with the server: {n, by_tool}. Absent log = zero calls.

    Zero is a real and interesting answer here, not a missing measurement: the prompt does
    not mention the tools, so an agent that never calls them has told us something about
    discoverability. The distinction that matters is between "ran, called nothing" and "the
    run never happened", and the caller knows which of those it has from `status`.
    """
    if not path.exists():
        return {"n": 0, "by_tool": {}}
    calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_tool: dict[str, int] = {}
    for c in calls:
        by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
    return {"n": len(calls), "by_tool": dict(sorted(by_tool.items()))}


def verify_case(case: str, *, results_path: Path = ARM_A) -> dict[str, Any]:
    """Read a seeded store back through the SHIPPED MCP tool body and check three things.

    Read back rather than trusted: this is the only check that exercises what the agent
    will actually receive, and it is the check that would catch a leak.
    """
    from reporadar.mcp_server import ranked_papers_payload

    db = case_db(case)
    if not db.exists():
        return {"case": case, "status": "missing"}
    expected = arm_picks(results_path, case)
    repo_dir = WORK_DIR / case
    with PaperStore(db) as store:
        payload = ranked_papers_payload(
            store,
            limit=50,
            repo_path=repo_dir,
            top_n=15,
            triage_threshold=2,
            rerank=True,
        )
    served = [p["arxiv_id"] for p in payload["papers"]]
    muted = {dedup_id(p["arxiv_id"]) for p in payload.get("muted", [])}
    # A', not A: the product's already-cited mute is real behaviour and the arm keeps it.
    # The invariant that has to hold is that nothing ELSE went missing.
    expected_ids = [dedup_id(p["arxiv_id"]) for p in expected]
    blob = json.dumps(payload)
    return {
        "case": case,
        "status": "ok",
        "n_arm_a": len(expected),
        "n_served": len(served),
        "n_muted_already_cited": len(muted),
        "serves_arm_a_minus_muted": {dedup_id(i) for i in served}
        == {i for i in expected_ids if i not in muted},
        "order_matches_arm_a": served == [p["arxiv_id"] for p in expected],
        "leaks_judge_fields": [f for f in FORBIDDEN_FIELDS if f in blob],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", action="store_true", help="build the per-case MCP stores ($0)")
    ap.add_argument("--verify", action="store_true", help="read the stores back ($0)")
    ap.add_argument("--case", help="comma-separated case names (default: every case arm A ran)")
    ap.add_argument("--arm-a", type=Path, default=ARM_A)
    args = ap.parse_args()

    rows = json.loads(args.arm_a.read_text(encoding="utf-8"))
    cases = args.case.split(",") if args.case else [r["case"] for r in rows]

    if args.seed:
        for name in cases:
            rec = seed_case(name, results_path=args.arm_a)
            write_config(name)  # the tokenless pair, for a manual run against one case
            print(f"  {name:<18} {rec['n_picks']:>2} picks -> {rec['db']}")
    if args.verify or not args.seed:
        bad = 0
        for name in cases:
            v = verify_case(name, results_path=args.arm_a)
            flag = ""
            if v["status"] != "ok":
                flag = "  MISSING"
            elif v["leaks_judge_fields"]:
                flag = f"  LEAKS {v['leaks_judge_fields']}"
            elif not v["serves_arm_a_minus_muted"]:
                flag = "  SET DIFFERS FROM ARM A"
            elif v["n_muted_already_cited"]:
                flag = f"  ({v['n_muted_already_cited']} muted: already cited by the repo)"
            elif not v["order_matches_arm_a"]:
                flag = "  (order differs within an llm_score group)"
            if flag.strip().startswith(("MISSING", "LEAKS", "SET")):
                bad += 1
            print(f"  {name:<18} served {v.get('n_served', 0):>2}/{v.get('n_arm_a', 0)}{flag}")
        if bad:
            print(f"\n{bad} case(s) unusable — do not run the arm.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
