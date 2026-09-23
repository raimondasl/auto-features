"""What does one repository actually cost to run?

The published figure, "~$0.01-0.02 per repository per run", has no derivation anywhere in
this repository. No run file records tokens, cost or spend for the Anonymous side. The only
price model in the tree is `evals/baseline.py`, and it prices the *comparator*. So the
headline cost comparison puts a measured baseline against an unmeasured system.

This measures it. Three call sites cost money per run:

* the **gate**, `claude-haiku-4-5`, once per candidate at `gate_depth` 50;
* **HyDE** hypothesis generation, once per repository;
* the **fine-scale rescore**, `gpt-4o-mini`, once per score-2 band paper.

Everything else is local: TF-IDF, BM25, the embedding encoder and the two-parameter logistic.

Prompts come from the shipped builders, never reimplemented here, so the token counts are the
token counts the product produces. Usage is read from the API response rather than estimated
from a tokenizer, so output tokens are real rather than bounded by `max_tokens`.

    uv run python evals/measure_cost.py --cases 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402

from anonymous.config import ProfilerConfig  # noqa: E402
from anonymous.profiler import profile_repo  # noqa: E402
from anonymous.triage import build_triage_prompt  # noqa: E402

EVALS = Path(__file__).resolve().parent
POOL = EVALS / ".work" / "pool-cut100"
OUT = EVALS / "cost_measured.json"

# List prices per million tokens, 2026-09. Recorded here so a future reader can tell which
# prices a figure was computed at, which is exactly what the published number could not say.
PRICES = {
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00, "source": "Anthropic list, 2026-09"},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60, "source": "OpenAI list, 2026-09"},
}

GATE_DEPTH = 50  # what the measured configuration gates
GATE_MAX_TOKENS = 200  # triage.score_actionability


def gate_prompts(case: str, limit: int) -> list[str]:
    """The real gate prompts for this repository, via the shipped builders."""
    repo = EVALS / ".work" / case
    profile = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=300))
    candidates = json.loads((POOL / f"{case}.json").read_text(encoding="utf-8"))["candidates"]
    return [build_triage_prompt(p, profile) for p in candidates[:limit]]


def measure_gate(cases: list[str], limit: int) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    model = "claude-haiku-4-5"
    tin = tout = calls = 0
    per_case: dict[str, dict] = {}
    for case in cases:
        c_in = c_out = 0
        for prompt in gate_prompts(case, limit):
            resp = client.messages.create(
                model=model,
                max_tokens=GATE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            c_in += resp.usage.input_tokens
            c_out += resp.usage.output_tokens
            calls += 1
        per_case[case] = {"calls": limit, "in": c_in, "out": c_out}
        tin += c_in
        tout += c_out
        print(f"  gate {case:<16} {limit:>3} calls  in={c_in:>7}  out={c_out:>5}")
    return {"model": model, "calls": calls, "in": tin, "out": tout, "per_case": per_case}


def measure_rescore(cases: list[str], per_repo: float) -> dict:
    """Rescore one band-sized sample per case, then scale to the measured band size."""
    import exp_finescale as ef
    from openai import OpenAI

    client = OpenAI()
    tin = tout = calls = 0
    for case in cases:
        repo = EVALS / ".work" / case
        profile = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=300))
        candidates = json.loads((POOL / f"{case}.json").read_text(encoding="utf-8"))["candidates"]
        for paper in candidates[:3]:
            common = {
                "repo": tb.repo_block(case),
                "title": paper.get("title") or "",
                "abstract": (paper.get("abstract") or "")[:1500],
            }
            resp = client.chat.completions.create(
                model=ef.MODEL,
                messages=[{"role": "user", "content": ef.SCALE_PROMPT.format(**common)}],
                temperature=0,
                max_tokens=4,
                logprobs=True,
                top_logprobs=20,
            )
            tin += resp.usage.prompt_tokens
            tout += resp.usage.completion_tokens
            calls += 1
        del profile
    return {
        "model": ef.MODEL,
        "calls": calls,
        "in": tin,
        "out": tout,
        "band_papers_per_repo": per_repo,
    }


def dollars(model: str, tin: int, tout: int) -> float:
    p = PRICES[model]
    return tin / 1_000_000 * p["in"] + tout / 1_000_000 * p["out"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=3, help="repositories to sample")
    ap.add_argument("--gate-limit", type=int, default=GATE_DEPTH)
    args = ap.parse_args()

    for key, who in (("ANTHROPIC_API_KEY", "gate"), ("OPENAI_API_KEY", "rescore")):
        if not os.environ.get(key):
            print(f"{key} is not set; needed for the {who}", file=sys.stderr)
            return 2

    cases = sorted(p.stem for p in POOL.glob("*.json"))[: args.cases]
    print(f"sampling {len(cases)} repositories: {', '.join(cases)}\n")

    gate = measure_gate(cases, args.gate_limit)
    # 315 band papers over 37 cases in the measured configuration.
    rescore = measure_rescore(cases, 315 / 37)

    n = len(cases)
    gate_per_repo = dollars(gate["model"], gate["in"] / n, gate["out"] / n)
    per_call_in = rescore["in"] / rescore["calls"]
    per_call_out = rescore["out"] / rescore["calls"]
    band = rescore["band_papers_per_repo"]
    rescore_per_repo = dollars(rescore["model"], per_call_in * band, per_call_out * band)

    total = gate_per_repo + rescore_per_repo
    summary = {
        "prices": PRICES,
        "sampled_cases": cases,
        "gate": gate,
        "rescore": rescore,
        "per_repository_usd": {
            "gate": round(gate_per_repo, 5),
            "rescore": round(rescore_per_repo, 5),
            "total_excl_hyde": round(total, 5),
        },
        "published_claim": "$0.01-0.02 per repository per run",
        "note": (
            "HyDE hypothesis generation is one further call per repository and is not "
            "included here. The gate dominates."
        ),
    }
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    tokens = f"{gate['in'] // n:,} in, {gate['out'] // n:,} out"
    print(f"\n  gate     ${gate_per_repo:.4f} / repo  ({tokens})")
    print(f"  rescore  ${rescore_per_repo:.4f} / repo  ({band:.1f} band papers)")
    print(f"  TOTAL    ${total:.4f} / repo, excluding HyDE")
    print(f"\n  published claim: $0.01-0.02 -> measured is {total / 0.015:.1f}x the midpoint")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
