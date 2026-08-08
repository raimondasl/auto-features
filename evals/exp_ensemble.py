"""E3 — ensemble vote fraction + verbalized P, the Anthropic-native calibration path.

The research pass's third experiment (evals/RESEARCH-score2-ranking.md §4.3), built on
the Simulated Annotators result (best published calibration construction: ECE 0.058-0.095
vs 0.217-0.374 single-shot). Per paper: N Haiku calls at the API's default temperature,
each under a DIFFERENT reviewer persona (prompt diversity is load-bearing — identical
votes share identical biases, the confident-wrong-consensus failure). Every call must
first state the strongest reason the paper is NOT actionable ("consider the alternative"
elicitation), then verdict + verbalized probability.

P-hat = mean over valid samples of (vote + verbalized/100)/2 (Avg-Conf). Pre-registered
policy: show iff P-hat >= 2/3. Success: pooled ECE <= 0.15 AND the raw (un-recalibrated)
2/3 policy beats show-all; linter's band paper falls below threshold. Kill: P-hat
clusters in 0.7-0.9 regardless of judge label; ECE > 0.3; band AUC <= 0.55.

    uv run python evals/exp_ensemble.py --testbed a b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402

from reporadar.llm_client import LLMError, complete  # noqa: E402

MODEL = "claude-haiku-4-5"

# Ten personas; the diversity is the mechanism, not decoration. Fixed globally.
PERSONAS = [
    "the repository's lead maintainer, triaging what is worth a branch this month",
    "a performance engineer who only cares about measurable wins on this codebase",
    "a skeptical senior reviewer whose default answer is no",
    "a staff engineer deciding what makes the quarterly roadmap",
    "a research engineer who has productionized many papers and knows most do not survive",
    "a test and reliability engineer worried about regressions and maintenance cost",
    "a pragmatic tech lead weighing implementation effort against expected benefit",
    "an open-source contributor looking for a concrete, well-scoped PR to write",
    "an API designer who cares whether the method fits this project's architecture",
    "an engineer on call for this system, interested only in changes that reduce real pain",
]

PROMPT = """\
You are {persona}. Decide whether the maintainers of the repository below should ACT on
this paper - integrate its method as a concrete improvement to this codebase. Topical
relevance is not enough; the bar is a change a maintainer would actually start.

# Repository
{repo}
# Candidate paper
Title: {title}
Abstract: {abstract}

First, state the single STRONGEST reason this paper is NOT actionable for this
repository. Then weigh it against the best reason it is, and answer.

Respond with ONLY a JSON object:
{{"counter": "<strongest reason it is NOT actionable>",
  "act": true|false,
  "p": <0-100, your probability that acting on this would genuinely improve this repo>}}"""


def one_vote(case: str, paper: tb.Paper, idx: int, cache_dir: Path) -> dict:
    slot = cache_dir / f"{case}_{paper.id.replace('/', '_')}_{idx}.json"
    if slot.is_file():
        return json.loads(slot.read_text(encoding="utf-8"))
    prompt = PROMPT.format(
        persona=PERSONAS[idx % len(PERSONAS)],
        repo=tb.repo_block(case),
        title=paper.title,
        abstract=paper.abstract[:1500],
    )
    cfg = SimpleNamespace(provider="claude", claude_model=MODEL, timeout=90)
    rec: dict = {}
    for attempt in range(3):
        try:
            raw = complete(prompt, cfg, max_tokens=400)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"no JSON in: {raw[:100]}")
            data = json.loads(match.group(0))
            rec = {"act": bool(data["act"]), "p": max(0, min(100, int(data["p"])))}
            break
        except (LLMError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            rec = {"error": f"{type(exc).__name__}: {exc}"}
            time.sleep(2 * (attempt + 1))
    if "act" in rec:  # cache only successes — a cached failure would be permanent
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def p_hat(votes: list[dict]) -> dict | None:
    valid = [v for v in votes if "act" in v]
    if not valid:
        return None
    vote_frac = sum(1 for v in valid if v["act"]) / len(valid)
    verbal = sum(v["p"] for v in valid) / len(valid) / 100.0
    return {
        "p_hat": (vote_frac + verbal) / 2,
        "vote_frac": vote_frac,
        "verbal_mean": verbal,
        "n_valid": len(valid),
    }


def papers_for(bed: str) -> dict[str, list[tb.Paper]]:
    if bed == "a":
        return {c: [p for p in b.papers if p.abstract] for c, b in tb.load_testbed_a().items()}
    if bed == "b":
        return {c: [p for p in b.admitted if p.abstract] for c, b in tb.load_testbed_b().items()}
    raise SystemExit(f"unknown testbed {bed!r} (E3 pre-registered A and B only)")


def summarize(bed: str, papers: dict[str, list[tb.Paper]], scored: dict) -> dict:
    p_scores = {c: {i: r["p_hat"] for i, r in per.items() if r} for c, per in scored.items()}
    summary: dict = {"testbed": bed, "model": MODEL, "n_personas": len(PERSONAS)}
    if bed == "a":
        bands = tb.load_testbed_a()
        summary["auc_band_judge2"] = tb.pooled_band_auc(bands, p_scores, target=2)
        summary["auc_band_judge3"] = tb.pooled_band_auc(bands, p_scores, target=3)
        probs, labels = [], []
        for c, per in papers.items():
            for p in per:
                v = p_scores.get(c, {}).get(p.id)
                if v is not None:
                    probs.append(v)
                    labels.append(p.actionable)
        summary["brier"] = tb.brier(probs, labels)
        summary["ece"] = tb.ece(probs, labels)
        summary["reliability"] = tb.reliability_table(probs, labels)
        baselines = tb.baseline_nets(bands)
        per_case = {}
        for case, band in bands.items():
            policy = tb.policy_net(band, p_scores.get(case, {}))
            per_case[case] = {
                "policy": policy,
                "show_all": baselines[case]["show_all"],
                "delta": policy - baselines[case]["show_all"],
            }
        summary["per_case"] = per_case
        summary["policy_mean"] = round(
            sum(v["policy"] for v in per_case.values()) / len(per_case), 3
        )
        summary["show_all_mean"] = round(
            sum(v["show_all"] for v in per_case.values()) / len(per_case), 3
        )
        summary["sign_test"] = tb.sign_test([v["delta"] for v in per_case.values()])
        # The pre-registered spot check: does linter's lone band paper fall below 2/3?
        linter = p_scores.get("linter", {})
        summary["linter_p_hats"] = {i: round(v, 3) for i, v in linter.items()}
    else:
        xs, ys = [], []
        for c, per in papers.items():
            for p in per:
                v = p_scores.get(c, {}).get(p.id)
                if v is not None:
                    xs.append(v)
                    ys.append(p.judge >= 3)
        summary["auc_judge3"] = tb.auc(xs, ys)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testbed", nargs="+", default=["a"], choices=["a", "b"])
    ap.add_argument("--votes", type=int, default=10)
    ap.add_argument("--case", help="single case, for a smoke run")
    ap.add_argument("--limit", type=int, help="max papers per case (smoke)")
    args = ap.parse_args()

    tb.load_env()
    for bed in args.testbed:
        papers = papers_for(bed)
        if args.case:
            papers = {args.case: papers[args.case]}
        cache_dir = tb.EXP / "cache" / "ensemble" / bed
        scored: dict[str, dict] = {}
        from concurrent.futures import ThreadPoolExecutor

        for case in sorted(papers):
            t0 = time.time()
            todo = papers[case][: args.limit]
            tasks = [(p, i) for p in todo for i in range(args.votes)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(
                    pool.map(lambda t, c=case, cd=cache_dir: one_vote(c, t[0], t[1], cd), tasks)
                )
            votes_of: dict[str, list[dict]] = {}
            for (p, _i), rec in zip(tasks, results, strict=True):
                votes_of.setdefault(p.id, []).append(rec)
            per: dict[str, dict | None] = {p.id: p_hat(votes_of[p.id]) for p in todo}
            scored[case] = per
            dt = time.time() - t0
            print(f"[{bed}:{case:10}] scored {len(per)} papers ({dt:.0f}s)", flush=True)
        summary = summarize(bed, papers, scored)
        out = tb.EXP / f"ensemble_{bed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "scored": scored}, indent=1), encoding="utf-8"
        )
        print(f"\n=== E3 {bed} ===")
        for k, v in summary.items():
            if k not in ("per_case", "reliability", "testbed", "model"):
                print(f"  {k}: {v}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
