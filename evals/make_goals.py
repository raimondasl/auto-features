"""Generate the goal statements for the stated-intent experiment (roadmap item 0).

Two size-based remedies for the thin-docs failure are already refuted — a similarity floor
on the dense search (the papers are close, the query is wrong) and a profile-information
floor (across the ablation the spread *between repos at one budget* is larger than the
movement between budgets: at 1500 chars `speech` scores −6.0 while `db` scores +9.0; and in
the real cohort precision *falls* as corpus grows, 108 ch → 1.00, 3,556 ch → 0.75). What is
left is not volume but **register**: documentation says what a project IS, and the paper
that would help says what it should ADOPT. A maintainer's stated goal is the one available
input already in the second register.

P8 measured stated wants fed to the **gate** at net@2 +57 against +95 — the worst arm in the
campaign — and concluded they belong in the *query*. This supplies that query input, and
`hyde.generate_hypotheses` appends it after the shared repo block so it structurally cannot
reach the gate or disturb the fine-scale map fitted to those bytes.

Three arms, and the difference between the last two is the whole design:

``control``
    No goal. Current behaviour.
``docs``
    A goal written from **exactly the bytes the system already has** —
    :func:`~reporadar.triage.repo_context_block`, nothing else — asked the improvement
    question. This is the arm that isolates *register* from *information*.
``blind``
    A model reads the **repository** — README at ``assemble_repo_context``'s 3,500-char
    budget rather than the profile's 300, plus a sample of **source the profiler never
    reads** (``scan_source`` is False) — and answers what maintainers would most want to
    improve. It is shown **no paper, no gold target, no judge verdict**.

``oracle``
    A model is shown the papers a judge already confirmed actionable for that case and asked
    to recover the goal statement that would have led someone there. **Deliberately leaky**,
    reported only as an unachievable ceiling — the same device as the calibration audit's
    oracle threshold.

**`docs` exists because `blind` alone cannot be attributed.** `blind` changes two things at
once: it asks a different *question* (a need, not an identity) and it sees strictly more
*information* (source the profiler never reads, and a README budget of 3,500 against the
profile's 300). Those imply different products — a register flip is one cheap LLM call,
while a source-code dependency means turning scanning on — so an experiment that moves both
learns which is true about neither. `docs` holds information fixed at exactly what the
pipeline already consumes and moves only the question. Read the three together:

* ``docs`` ≈ ``blind`` → the win is the **question**, and it ships as one cheap call.
* ``docs`` ≈ ``control`` → the win is the **source code**, a larger and different change.

The oracle is generated rather than hand-written on purpose. Whoever ran the earlier
experiments has already read these cases' judged papers, so anything they author carries an
unmeasurable amount of that knowledge; a scripted oracle has its leak as a *documented
input* instead, and re-derives identically.

    uv run python evals/make_goals.py --arm blind    # ~$0.05
    uv run python evals/make_goals.py --arm oracle   # ~$0.05, leaky by construction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import WORK_DIR, assemble_repo_context, load_benchmark, profile_case_repo  # noqa: E402
from run_judge_eval import ENV_KEYS, RESULTS_DIR, load_dotenv  # noqa: E402

from reporadar.llm_client import complete  # noqa: E402
from reporadar.triage import repo_context_block  # noqa: E402

EVALS = Path(__file__).resolve().parent
GOALS_DIR = EVALS / "goals"
MODEL = "claude-haiku-4-5"

BLIND_PROMPT = """\
You are shown a software repository. In ONE sentence of at most 40 words, state what its
maintainers would most want to IMPROVE about it — a capability or quality the project does
not yet have, or does poorly.

Write what the project NEEDS, not what it already does. "A key-value store with a memcached
protocol" describes what it is; "reduce write amplification during compaction under
write-heavy workloads" states what it needs. Only the second is useful here.

Do not mention papers, research, or literature. Do not name the repository.

Repository:
{context}

Respond with ONLY the sentence.
"""

ORACLE_PROMPT = """\
Below are research papers that an expert judge confirmed would genuinely improve a specific
software repository, plus a description of that repository.

In ONE sentence of at most 40 words, write the goal statement a maintainer could have
written — BEFORE seeing any of these papers — that would have led a literature search to
them. State what the project needs to improve, in the maintainer's own terms.

Do not mention the papers, their titles, their methods, or any research vocabulary that
only appears because you were shown them. If your sentence could only have been written by
someone who had already read these papers, it is wrong. Do not name the repository.

Repository:
{context}

Papers the judge confirmed actionable:
{papers}

Respond with ONLY the sentence.
"""


def _cfg(keys: dict[str, str]) -> Any:
    return SimpleNamespace(
        provider="claude",
        claude_api_key=keys.get("ANTHROPIC_API_KEY", ""),
        claude_model=MODEL,
        timeout=120,
    )


def source_sample(repo_dir: Path, *, max_files: int = 12, per_file: int = 800) -> str:
    """A sample of real source, which the profiler never reads.

    The blind arm is allowed strictly more than the product currently uses — that is the
    point of a ceiling. A goal has to come from somewhere, and code is the one description
    of a thin-docs repository that is always present.
    """
    exts = (".py", ".rs", ".go", ".cpp", ".c", ".java", ".ts", ".js")
    out: list[str] = []
    for path in sorted(repo_dir.rglob("*")):
        if len(out) >= max_files:
            break
        if path.is_file() and path.suffix in exts and ".git" not in path.parts:
            try:
                body = path.read_text("utf-8", "ignore")[:per_file]
            except OSError:
                continue
            out.append(f"--- {path.relative_to(repo_dir)}\n{body}")
    return "\n".join(out)


def actionable_papers(case: str, run_file: Path) -> list[dict[str, Any]]:
    """Papers the judge scored >= 2 for this case in a completed run. The oracle's leak."""
    run = {r["case"]: r for r in json.loads(run_file.read_text(encoding="utf-8"))}
    rec = run.get(case)
    if not rec:
        return []
    seen: dict[str, dict[str, Any]] = {}
    for group in rec["returned"].values():
        for p in group:
            if (p.get("judge_score") or 0) >= 2:
                seen.setdefault(p["arxiv_id"].split("v")[0], p)
    return list(seen.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("docs", "blind", "oracle"), required=True)
    ap.add_argument("--cases", default="thin-lang,thin-kv,thin-gnn,compiler,storage,graph")
    ap.add_argument("--run", default="judge-gpt-5.5-20260809T181850Z.json", help="oracle source")
    args = ap.parse_args()

    load_dotenv(EVALS / ".env")
    import os

    keys = {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}
    if "ANTHROPIC_API_KEY" not in keys:
        raise SystemExit("ANTHROPIC_API_KEY required; nothing was called")

    bench = {c["name"]: c for c in load_benchmark()["cases"]}
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = set(wanted) - set(bench)
    if unknown:
        raise SystemExit(f"--cases not in the benchmark: {sorted(unknown)}")

    GOALS_DIR.mkdir(exist_ok=True)
    out_path = GOALS_DIR / f"{args.arm}.json"
    goals: dict[str, str] = {}
    if out_path.is_file():
        goals = json.loads(out_path.read_text(encoding="utf-8"))

    for case in wanted:
        if case in goals:
            print(f"  {case:11} cached")
            continue
        repo_dir = WORK_DIR / case
        if not repo_dir.is_dir():
            print(f"  {case:11} !! not cloned — run the benchmark first; SKIPPED")
            continue
        context = assemble_repo_context(repo_dir)
        if args.arm == "docs":
            # Exactly what the gate, the rescore and the hypothesis prompt already see —
            # no source, no longer README. Only the question changes.
            prompt = BLIND_PROMPT.format(context=repo_context_block(profile_case_repo(repo_dir)))
        elif args.arm == "blind":
            context = f"{context}\n\n## Source sample\n{source_sample(repo_dir)}"
            prompt = BLIND_PROMPT.format(context=context[:14000])
        else:
            papers = actionable_papers(case, RESULTS_DIR / args.run)
            if not papers:
                print(f"  {case:11} !! no judge-confirmed papers in {args.run}; SKIPPED")
                continue
            listing = "\n".join(f"- {p['title']}" for p in papers)
            prompt = ORACLE_PROMPT.format(context=context[:10000], papers=listing)
        goal = " ".join(complete(prompt, _cfg(keys), max_tokens=200).split())
        goals[case] = goal
        print(f"  {case:11} {goal}")
        out_path.write_text(json.dumps(goals, indent=1, sort_keys=True), encoding="utf-8")

    print(f"\nWrote {out_path} ({len(goals)} goals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
