"""Is the "lacks" prompt wrong, or is the benchmark asking it the wrong question?

P2 concluded that "lacks" phrases suffer a *different-target failure*: they name a coherent
research agenda that is not what the judge rewards. On `rl` they proposed offline RL,
multi-agent and imitation learning while the judged targets were Double Q-Learning and
Prioritized Experience Replay — refinements of what stable-baselines3 already implements.

That conclusion assumed the judge is the arbiter. But the judge's rubric says
**"genuinely IMPROVE"**, and scores 3 for a paper that "directly addresses a known limitation
or core capability of this repository". A paper adding offline RL to a library that has none
is not improving a known limitation of the existing code — it is *extending the project's
scope*. The rubric would score it low even if a maintainer would want it.

So "lacks retrieves badly" and "lacks retrieves EXTENSIONS, which this benchmark does not
measure" predict the same P2 numbers. This separates them:

    uv run python evals/extend_vs_improve.py            # ~$3 of judge calls

Two rubrics over the SAME papers, same repo context, same model:

  IMPROVE  the shipped judge rubric, verbatim — refine what the repo already does
  EXTEND   would this let the repo do something it currently CANNOT do? Deliberately
           scores DOWN a paper that merely refines an existing component, so the two
           rubrics can dissociate rather than one being a looser version of the other.

Two paper groups per case:

  lacks    the top-8 the "lacks" query surfaces from the citation-hop pool
  targets  the known-good papers (baseline picks the judge scored >=2 under IMPROVE)

The design is a 2x2 and the interesting cell is the dissociation. If `lacks` papers score
low on IMPROVE and high on EXTEND while `targets` do the reverse, the "lacks" prompt is
answering a real question the benchmark never asked, and belongs behind an explicit
"extend this project" mode rather than in the improvement pipeline. If `lacks` papers score
low on BOTH, P2's conclusion stands unchanged and the prompt is simply weak.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import diagnose_query_generation as qg  # noqa: E402
import judge as judge_mod  # noqa: E402
from build_hop_pool import resolve_targets  # noqa: E402
from gap_match import LACKS_CACHE, bm25, stem_tokens  # noqa: E402
from harness import WORK_DIR, assemble_repo_context  # noqa: E402

EVALS = Path(__file__).resolve().parent
POOL_DIR = EVALS / ".work" / "hop_pool"
OUT = EVALS / ".work" / "extend_vs_improve.json"
TOP_N = 8

EXTEND_RUBRIC = """\
You are a neutral senior research engineer judging whether a research paper would let a
software repository do something it currently CANNOT do — a NEW CAPABILITY, not a better
version of an existing one.

Score the paper for THIS repository on a 0-3 scale:
  0 = unrelated, or the repository already does this.
  1 = a new capability, but one this project would reasonably consider out of scope.
  2 = a capability the project plausibly SHOULD have and currently lacks, with a
      concrete implementation path.
  3 = a capability whose absence is a visible gap in this project — users would
      reasonably expect it, and the paper provides a strong, specific way to add it.

IMPORTANT: a paper that merely REFINES something the repository already implements
(a better optimiser for an algorithm it has, a faster version of an existing component)
scores AT MOST 1 here, however good it is. This rubric measures new reach, not quality.

Ground your justification only in the provided abstract — do not invent findings.

Respond with a JSON object with exactly these keys:
  "score": one of 0, 1, 2, 3
  "justification": one sentence
  "proposed_change": one concrete capability this would add (empty string if score < 2)
"""


def lacks_top(case: str, cache: dict, n: int) -> list[dict]:
    """Top-n papers the 'lacks' query surfaces from this case's hop pool."""
    path = POOL_DIR / f"{case}.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    wt = [r for r in rows if (cache.get(r["id"]) or {}).get("abstract")]
    lacks = json.loads(LACKS_CACHE.read_text(encoding="utf-8")).get(case) or []
    if not wt or not lacks:
        return []
    corpus = [stem_tokens(f"{cache[r['id']]['title']} {cache[r['id']]['abstract']}") for r in wt]
    scores = bm25(corpus, stem_tokens(" ".join(lacks)))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [
        {
            "arxiv_id": wt[i]["id"],
            "title": cache[wt[i]["id"]]["title"],
            "abstract": cache[wt[i]["id"]]["abstract"],
        }
        for i in order
    ]


def random_sample(case: str, cache: dict, n: int, exclude: set[str]) -> list[dict]:
    """A uniform random sample of the pool — the control that makes the others readable.

    Without it, "26 of 40 lacks papers score >=2" is uninterpretable: it could mean the
    query retrieves well, or that the judge rates most on-topic papers >=2. Those imply
    opposite conclusions and the difference is one cheap sample.

    Seeded, so the control is the same set on a re-run.
    """
    path = POOL_DIR / f"{case}.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    pool = [
        r for r in rows if (cache.get(r["id"]) or {}).get("abstract") and r["id"] not in exclude
    ]
    rng = random.Random(1234)
    picked = rng.sample(pool, min(n, len(pool)))
    return [
        {
            "arxiv_id": r["id"],
            "title": cache[r["id"]]["title"],
            "abstract": cache[r["id"]]["abstract"],
        }
        for r in picked
    ]


def score_group(
    case: str, ctx: str, papers: list[dict], rubric: str, tag: str, model: str
) -> list[dict]:
    """Judge every paper under one rubric. Swapping RUBRIC is the only variable.

    `judge_paper` keys its cache on a hash of the rubric plus the repo context, so the two
    rubrics cannot collide in the cache and an edit to either re-judges rather than serving
    a stale verdict.
    """
    out = []
    original = judge_mod.RUBRIC
    judge_mod.RUBRIC = rubric
    try:
        for p in papers:
            try:
                # use_cache=False is LOAD-BEARING, not an optimisation. judge_paper keys its
                # cache file on (model, repo, paper_id) and NOT on the rubric — the rubric
                # only lands inside the file as `_prompt_hash`. So writing an EXTEND verdict
                # overwrites that paper's IMPROVE verdict in the shared gold cache. This
                # script did exactly that on its first real run and knocked 9 known targets
                # below the >=2 threshold, taking `rag` from 5 targets to 0, because
                # `diagnose_pool.actionable_baseline_ids` reads the score without checking
                # the hash. Caught by tests/test_eval_hop_pool.py; restored by re-judging.
                v = judge_mod.judge_paper(case, ctx, p, model=model, use_cache=False)
            except Exception as exc:  # noqa: BLE001
                print(f"      ! {p['arxiv_id']} {tag} failed: {exc}")
                continue
            out.append({"id": p["arxiv_id"], "title": p["title"], **v})
    finally:
        judge_mod.RUBRIC = original
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="rl,peft,cv,rag,graph")
    ap.add_argument("--model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()
    qg._load_env()  # judge.py reads OPENAI_API_KEY from the environment, not from disk

    cache = json.loads((POOL_DIR / "_meta_cache.json").read_text(encoding="utf-8"))
    targets = resolve_targets()
    rows = []

    for case in args.cases.split(","):
        repo = WORK_DIR / case
        if not repo.is_dir():
            print(f"[{case}] no clone — skipping")
            continue
        ctx = assemble_repo_context(repo)
        lacks_papers = lacks_top(case, cache, TOP_N)
        tgt_papers = [
            {"arxiv_id": t, "title": cache[t]["title"], "abstract": cache[t]["abstract"]}
            for t in targets.get(case, [])
            if t in cache and cache[t].get("abstract")
        ]
        if not lacks_papers:
            print(f"[{case}] no lacks retrievals — skipping")
            continue
        print(f"[{case}] {len(lacks_papers)} lacks papers, {len(tgt_papers)} targets", flush=True)
        seen = {p["arxiv_id"] for p in lacks_papers} | {p["arxiv_id"] for p in tgt_papers}
        rand_papers = random_sample(case, cache, TOP_N, seen)
        groups = (("lacks", lacks_papers), ("targets", tgt_papers), ("random", rand_papers))
        for group, papers in groups:
            for rubric_name, rubric in (("improve", judge_mod.RUBRIC), ("extend", EXTEND_RUBRIC)):
                verdicts = score_group(case, ctx, papers, rubric, rubric_name, args.model)
                for v in verdicts:
                    rows.append({"case": case, "group": group, "rubric": rubric_name, **v})
                if verdicts:
                    mean = sum(v["score"] for v in verdicts) / len(verdicts)
                    hi = sum(1 for v in verdicts if v["score"] >= 2)
                    print(
                        f"    {group:8} x {rubric_name:8} mean={mean:.2f}  "
                        f">=2: {hi}/{len(verdicts)}",
                        flush=True,
                    )

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # A verdict computed over no verdicts is not a null result, it is a broken run.
    # The first version of this script judged nothing (no API key) and still printed
    # ">>> NO DISSOCIATION ... P2 stands." — a confident conclusion from zero data.
    expected = {(g, r) for g in ("lacks", "targets", "random") for r in ("improve", "extend")}
    have = {(r["group"], r["rubric"]) for r in rows}
    if expected - have:
        missing = ", ".join(f"{g}x{r}" for g, r in sorted(expected - have))
        print(f"\n!! NO VERDICT — empty cells: {missing}. {len(rows)} verdicts total.")
        print("   Fix the run before reading anything into this.")
        return 1
    print("\n=== 2x2: mean score (and fraction scoring >=2) ===")
    print(f"{'':10} {'IMPROVE':>18} {'EXTEND':>18}")
    for group in ("lacks", "targets", "random"):
        cells = []
        for rubric in ("improve", "extend"):
            sel = [r for r in rows if r["group"] == group and r["rubric"] == rubric]
            if not sel:
                cells.append("       n/a")
                continue
            m = sum(r["score"] for r in sel) / len(sel)
            hi = sum(1 for r in sel if r["score"] >= 2)
            cells.append(f"{m:6.2f} ({hi:2}/{len(sel):2})")
        print(f"{group:10} {cells[0]:>18} {cells[1]:>18}")

    def cell(g: str, r: str) -> float:
        sel = [x for x in rows if x["group"] == g and x["rubric"] == r]
        return sum(x["score"] for x in sel) / len(sel) if sel else 0.0

    li, le = cell("lacks", "improve"), cell("lacks", "extend")
    ti, te = cell("targets", "improve"), cell("targets", "extend")
    print(f"\nlacks   improve {li:.2f} -> extend {le:.2f}   ({le - li:+.2f})")
    print(f"targets improve {ti:.2f} -> extend {te:.2f}   ({te - ti:+.2f})")
    if le - li > 0.5 and te - ti < 0:
        print("\n>>> DISSOCIATION: 'lacks' retrieves EXTENSIONS the improvement rubric cannot see.")
    elif le <= li:
        print("\n>>> NO DISSOCIATION: 'lacks' papers are not extensions either. P2 stands.")
    else:
        print("\n>>> PARTIAL: both groups rise under EXTEND — it may just be a looser rubric.")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
