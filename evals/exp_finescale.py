"""E2 — fine-scale distributional pointwise scoring via OpenAI logprobs.

The research pass's second experiment (evals/RESEARCH-score2-ranking.md §4.2).
Hypothesis: the score-2 pile-up is partly a quantization artifact. Two probes, both
gpt-4o-mini with logprobs (Anthropic exposes none):

(a) an anchored 0-9 scale — anchors written from the shipped rubric's own band text, not
    from benchmark failures — read as the EXPECTATION over the digit-token distribution
    (G-Eval / TrustJudge mechanism). 0-9 rather than the doc's 0-10 so the score is one
    token and the distribution is directly readable.
(b) brief forced weighing of the strongest reason NOT to act, then `ANSWER: true|false`;
    P(actionable) = normalized p("true") at the answer token (the Rank1 mechanism).

Pre-registered: success = pooled within-band AUC >= 0.65; promotion to calibration
candidate iff Brier <= 0.22 and the P>=2/3 policy beats show-all. Kill: the modal score
token carries p > 0.9 for > 80% of papers (the same collapse in finer clothes) or
AUC <= 0.55. Validity landmine: the judge is GPT-5.5 — any win must survive the Sonnet
second-judge cross-check before being believed (same-family bias).

    uv run python evals/exp_finescale.py --testbed a a300 b c

**The `haiku` arm** (added after E2 won, to answer whether shipping needs OpenAI at all):
the same 0-9 prompt sent N times to Haiku at the API's default temperature, with the
MEAN of the sampled digits standing in for the logprob expectation. Same prompt, same
estimand, Monte-Carlo estimator instead of an exact one — so the comparison isolates
what the logprobs buy. Its resolution is 1/N; the exact reading is continuous, which is
the specific thing at issue.

    uv run python evals/exp_finescale.py --arm haiku --samples 10 --testbed a a300
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402

MODEL = "gpt-4o-mini"

SCALE_PROMPT = """\
You are a senior research engineer scoring whether a paper can be used to IMPROVE a
specific software repository - not merely whether it is on a related topic. Score 0-9:
  0-1 = unrelated or not applicable to this repository.
  2-3 = same general topic, but no concrete, actionable improvement to this code.
  4-6 = proposes a method that could plausibly be integrated to improve this repository,
        with a concrete implementation path. Higher = clearer path, stronger evidence.
  7-9 = directly addresses a known limitation or core capability of this repository with
        a strong, specific, implementable improvement. Higher = more specific, stronger.
Use the full scale - distinguish shades within each band, and be strict: a low score is
the correct answer when the paper is not useful to this code.

# Repository
{repo}
# Candidate paper
Title: {title}
Abstract: {abstract}

Respond with ONLY a single digit 0-9."""

VERDICT_PROMPT = """\
You are a senior research engineer deciding whether the maintainers of one specific
software repository should act on a paper - integrate its method as a concrete
improvement to this codebase. Topical relevance is not enough.

# Repository
{repo}
# Candidate paper
Title: {title}
Abstract: {abstract}

In 2-3 sentences, weigh the STRONGEST reason this paper is NOT actionable for this
repository against the strongest reason it is. Then answer on the final line with
exactly `ANSWER: true` or `ANSWER: false` - true meaning maintainers should act on it."""


def _client():  # type: ignore[no-untyped-def]
    from openai import OpenAI

    return OpenAI()


def _digit_expectation(logprob_content: list) -> tuple[float, float] | None:
    """(expectation over digit tokens, modal digit prob) from the first digit token."""
    for tok in logprob_content:
        if tok.token.strip().isdigit():
            probs: dict[int, float] = {}
            for alt in tok.top_logprobs:
                text = alt.token.strip()
                if text.isdigit() and 0 <= int(text) <= 9:
                    probs[int(text)] = probs.get(int(text), 0.0) + math.exp(alt.logprob)
            total = sum(probs.values())
            if total <= 0:
                return None
            exp = sum(d * p for d, p in probs.items()) / total
            return exp, max(probs.values()) / total
    return None


def _p_true(logprob_content: list) -> float | None:
    """Normalized p(true) at the final true/false answer token."""
    for tok in reversed(logprob_content):
        text = tok.token.strip().lower()
        if text in ("true", "false"):
            p = {"true": 0.0, "false": 0.0}
            for alt in tok.top_logprobs:
                a = alt.token.strip().lower()
                if a in p:
                    p[a] += math.exp(alt.logprob)
            total = p["true"] + p["false"]
            return p["true"] / total if total > 0 else None
    return None


HAIKU_MODEL = "claude-haiku-4-5"


def _sampled_digit(raw: str) -> int | None:
    """The first 0-9 digit in a Haiku reply, or None if it did not answer as asked."""
    m = re.search(r"\d", raw)
    if not m:
        return None
    d = int(m.group(0))
    return d if 0 <= d <= 9 else None


def score_paper_haiku(case: str, paper: tb.Paper, samples: int, cache_dir: Path) -> dict:
    """Monte-Carlo stand-in for the logprob expectation: N draws, take the mean.

    Anthropic exposes no logprobs, so the only way to see the score *distribution* is to
    sample it. Resolution is 1/N against the exact reading's continuous one — which is
    precisely the quantity this arm exists to measure.
    """
    slot = cache_dir / f"{case}_{paper.id.replace('/', '_')}.json"
    if slot.is_file():
        cached = json.loads(slot.read_text(encoding="utf-8"))
        if len(cached.get("draws", [])) >= samples:
            return cached
    from types import SimpleNamespace

    from reporadar.llm_client import LLMError, complete

    cfg = SimpleNamespace(provider="claude", claude_model=HAIKU_MODEL, timeout=60)
    prompt = SCALE_PROMPT.format(
        repo=tb.repo_block(case), title=paper.title, abstract=paper.abstract[:1500]
    )

    def one(_i: int) -> int | None:
        for attempt in range(3):
            try:
                return _sampled_digit(complete(prompt, cfg, max_tokens=8))
            except LLMError:
                time.sleep(2 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        draws = [d for d in pool.map(one, range(samples)) if d is not None]
    rec: dict = {"draws": draws}
    if draws:
        rec["exp09"] = sum(draws) / len(draws)
        # The same degeneracy probe as the logprob arm: how concentrated is the draw?
        rec["modal_p"] = max(draws.count(d) for d in set(draws)) / len(draws)
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def score_paper(client, case: str, paper: tb.Paper, cache_dir: Path) -> dict:  # type: ignore[no-untyped-def]
    slot = cache_dir / f"{case}_{paper.id.replace('/', '_')}.json"
    if slot.is_file():
        return json.loads(slot.read_text(encoding="utf-8"))
    rec: dict = {}
    common = {"repo": tb.repo_block(case), "title": paper.title, "abstract": paper.abstract[:1500]}
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": SCALE_PROMPT.format(**common)}],
            temperature=0,
            max_tokens=4,
            logprobs=True,
            top_logprobs=20,
        )
        got = _digit_expectation(resp.choices[0].logprobs.content)
        if got:
            rec["exp09"], rec["modal_p"] = got
    except Exception as exc:  # noqa: BLE001
        rec["scale_error"] = f"{type(exc).__name__}: {exc}"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": VERDICT_PROMPT.format(**common)}],
            temperature=0,
            max_tokens=250,
            logprobs=True,
            top_logprobs=20,
        )
        rec["p_true"] = _p_true(resp.choices[0].logprobs.content)
    except Exception as exc:  # noqa: BLE001
        rec["verdict_error"] = f"{type(exc).__name__}: {exc}"
    if "exp09" in rec or rec.get("p_true") is not None:  # cache only usable results
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def papers_for(bed: str) -> dict[str, list[tb.Paper]]:
    """Which papers get scored: every labelled shown/band paper with gate-time text."""
    if bed == "a":
        return {c: [p for p in b.papers if p.abstract] for c, b in tb.load_testbed_a().items()}
    if bed == "a300":
        return {c: [p for p in b.papers if p.abstract] for c, b in tb.load_testbed_a300().items()}
    if bed == "b":
        return {c: [p for p in b.admitted if p.abstract] for c, b in tb.load_testbed_b().items()}
    if bed == "c":
        c = tb.load_testbed_c()
        out: dict[str, list[tb.Paper]] = {}
        for p in c["gate_full_pool"] + c["label_pool"]:
            if p.abstract:
                out.setdefault(p.case, []).append(p)
        # the two files can overlap on (case, id); keep one
        return {k: list({p.id: p for p in v}.values()) for k, v in out.items()}
    raise SystemExit(f"unknown testbed {bed!r}")


def summarize(bed: str, papers: dict[str, list[tb.Paper]], scored: dict) -> dict:
    exp_scores = {
        c: {i: r["exp09"] for i, r in per.items() if "exp09" in r} for c, per in scored.items()
    }
    p_scores = {
        c: {i: r["p_true"] for i, r in per.items() if r.get("p_true") is not None}
        for c, per in scored.items()
    }
    summary: dict = {"testbed": bed, "model": MODEL}

    # Kill-condition probe: is the digit distribution degenerate?
    modal = [r["modal_p"] for per in scored.values() for r in per.values() if "modal_p" in r]
    if modal:
        summary["modal_p_gt_090"] = round(sum(1 for m in modal if m > 0.9) / len(modal), 3)

    if bed in ("a", "a300"):
        bands = tb.load_testbed_a() if bed == "a" else tb.load_testbed_a300()
        for name, scores in (("exp09", exp_scores), ("p_true", p_scores)):
            summary[f"auc_band_judge2_{name}"] = tb.pooled_band_auc(bands, scores, target=2)
            summary[f"auc_band_judge3_{name}"] = tb.pooled_band_auc(bands, scores, target=3)
        # Calibration of p_true over ALL shown labelled papers (not just the band).
        probs, labels = [], []
        for c, per in papers.items():
            for p in per:
                v = p_scores.get(c, {}).get(p.id)
                if v is not None:
                    probs.append(v)
                    labels.append(p.actionable)
        summary["brier_p_true"] = tb.brier(probs, labels)
        summary["ece_p_true"] = tb.ece(probs, labels)
        summary["reliability_p_true"] = tb.reliability_table(probs, labels)
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
    else:
        # Pooled ordering metrics only (B: judge==3 target; C: judge>=2 in the wild).
        target = 3 if bed == "b" else 2
        for name, scores in (("exp09", exp_scores), ("p_true", p_scores)):
            xs, ys = [], []
            for c, per in papers.items():
                for p in per:
                    v = scores.get(c, {}).get(p.id)
                    if v is not None:
                        xs.append(v)
                        ys.append(p.judge >= target)
            summary[f"auc_judge{target}_{name}"] = tb.auc(xs, ys)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testbed", nargs="+", default=["a"], choices=["a", "a300", "b", "c"])
    ap.add_argument(
        "--arm",
        default="openai",
        choices=["openai", "haiku"],
        help="openai = exact logprob expectation; haiku = N-sample Monte-Carlo mean",
    )
    ap.add_argument("--samples", type=int, default=10, help="draws per paper in the haiku arm")
    ap.add_argument("--case", help="single case, for a smoke run")
    ap.add_argument("--limit", type=int, help="max papers per case (smoke)")
    args = ap.parse_args()

    tb.load_env()
    client = _client() if args.arm == "openai" else None
    suffix = "" if args.arm == "openai" else f"_haiku{args.samples}"
    for bed in args.testbed:
        papers = papers_for(bed)
        if args.case:
            papers = {args.case: papers[args.case]}
        cache_dir = tb.EXP / "cache" / f"finescale{suffix}" / bed
        scored: dict[str, dict[str, dict]] = {}
        t0 = time.time()

        for case in sorted(papers):
            todo = papers[case][: args.limit]
            with ThreadPoolExecutor(max_workers=8) as pool:
                if args.arm == "openai":
                    recs = list(
                        pool.map(
                            lambda p, c=case, cd=cache_dir: score_paper(client, c, p, cd), todo
                        )
                    )
                else:
                    recs = list(
                        pool.map(
                            lambda p, c=case, cd=cache_dir: score_paper_haiku(
                                c, p, args.samples, cd
                            ),
                            todo,
                        )
                    )
            scored[case] = dict(zip((p.id for p in todo), recs, strict=True))
            print(f"[{bed}:{case:10}] scored {len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        summary = summarize(bed, papers, scored)
        summary["arm"] = args.arm
        if args.arm == "haiku":
            summary["samples"] = args.samples
        out = tb.EXP / f"finescale{suffix}_{bed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "scored": scored}, indent=1), encoding="utf-8"
        )
        print(f"\n=== E2[{args.arm}] {bed} ({time.time() - t0:.0f}s) ===")
        for k, v in summary.items():
            if k not in ("per_case", "reliability_p_true", "testbed", "model"):
                print(f"  {k}: {v}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
