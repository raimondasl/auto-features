"""E4 — round-robin pairwise comparison with Bradley-Terry aggregation and anchors.

The research pass's fourth experiment (evals/RESEARCH-score2-ranking.md §4.4).
Hypothesis: comparative utility judgments carry the within-band signal absolute scores
discard (PRP +8 nDCG over pointwise; PairS's largest gains in exactly the weak-judge
regime). Both-order querying is mandatory plumbing, not an option — Claude-family
position bias has measured swap-consistency as low as 23.8%.

Per case: every pair of admitted papers, BOTH presentation orders, Haiku, evidence-first
comparative prompt. The two orders disagreeing = a tie (half-win each). Bradley-Terry
strengths by MLE (Zermelo iterations). The anchored arm adds three fixed-template
reference "papers" instantiated from the repo's own keyword profile — clearly-actionable
/ borderline (topical survey, nothing implementable) / off-topic — and the pre-registered
policy shows a band paper iff P_BT(beats the borderline anchor) >= 2/3.

Success: pooled within-band AUC >= 0.70 earns the ordering-backbone role even if
anchoring fails; the anchored policy graduates only if it beats show-all. Kill:
swap-inconsistency > 45% of pairs; anchor scale unstable under paraphrase.

Bands larger than 25 papers (Testbed B's peft) get a seeded random sample of unordered
pairs instead of the full round-robin, logged as such — no silent caps.

    uv run python evals/exp_pairwise.py --testbed a
    uv run python evals/exp_pairwise.py --testbed b --no-anchors
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
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
MAX_UNORDERED_PAIRS = 350  # full round-robin above this samples (seeded); logged loudly

PROMPT = """\
Two candidate papers for one specific software repository. Decide which one the
repository's maintainers should act on FIRST - the stronger concrete, implementable
improvement to THIS codebase. Topical relevance is not enough; judge which paper a
competent maintainer would actually start a branch for.

# Repository
{repo}
# Paper A
Title: {title_a}
Abstract: {abstract_a}

# Paper B
Title: {title_b}
Abstract: {abstract_b}

For the winner, name the specific repository component it improves and the concrete
change a maintainer would make. Respond with ONLY a JSON object:
{{"winner": "A" or "B", "component": "<component>", "change": "<change>"}}"""

# Fixed global templates, instantiated only with the repo's own profile keywords.
# The borderline anchor IS the calibration point: a topically-adjacent survey with
# nothing implementable — the rubric's score-1 archetype sitting just below the bar.
ANCHORS = {
    "anchor_high": (
        "A drop-in method for improving {kw} systems",
        "We present a technique that directly addresses a known limitation of {kw} "
        "systems. The method is a drop-in replacement for a core component, requires no "
        "architectural changes, and ships with an open-source reference implementation. "
        "On standard benchmarks it improves the primary quality metric by 18-31% at equal "
        "cost, with ablations isolating the gain to the proposed change. We document the "
        "exact integration path for existing codebases, including migration steps and "
        "failure modes observed in three production deployments.",
    ),
    "anchor_borderline": (
        "A survey of recent advances in {kw}",
        "We survey recent developments in {kw}, organizing the literature into a "
        "taxonomy of approaches and summarizing reported results across benchmarks. We "
        "discuss open challenges and promising directions for future work. The survey "
        "covers 148 papers and provides a comparative analysis of their assumptions, "
        "though we propose no new method and report no new experiments.",
    ),
    "anchor_low": (
        "Spectral properties of quasi-periodic tilings under thermal stress",
        "We analyze the spectral decomposition of quasi-periodic tilings subjected to "
        "thermal stress gradients, deriving closed-form expressions for the deformation "
        "modes of Penrose-type lattices. Numerical simulations confirm the predicted "
        "phase transitions at critical temperature ratios. Applications to materials "
        "science and crystallography are discussed.",
    ),
}

# One fixed paraphrase of the borderline anchor, for the pre-registered stability check.
BORDERLINE_PARAPHRASE = (
    "Recent progress in {kw}: a systematic review",
    "This article systematically reviews the state of the art in {kw}, categorizing "
    "published approaches and tabulating their reported benchmark results. We identify "
    "open problems and sketch directions future research might take. 148 works are "
    "covered in a comparative discussion of assumptions and evaluation practice; no new "
    "method is introduced and no new experiments are run.",
)


def anchor_papers(case: str, paraphrase: bool = False) -> list[tb.Paper]:
    """The three anchors instantiated with the repo's top keywords."""
    import re as _re

    block = tb.repo_block(case)
    m = _re.search(r"Key topics: (.+)", block)
    kw = ", ".join((m.group(1) if m else "software").split(", ")[:3])
    out = []
    for aid, (title, abstract) in ANCHORS.items():
        if paraphrase and aid == "anchor_borderline":
            title, abstract = BORDERLINE_PARAPHRASE
        out.append(
            tb.Paper(
                case=case,
                id=aid,
                title=title.format(kw=kw),
                abstract=abstract.format(kw=kw),
                judge=-1,
                gate=-1,
            )
        )
    return out


def one_comparison(case: str, a: tb.Paper, b: tb.Paper, cache_dir: Path) -> str | None:
    """'A' | 'B' | None(failure) for papers presented in this order. Cached."""
    slot = cache_dir / f"{case}__{a.id.replace('/', '_')}__{b.id.replace('/', '_')}.json"
    if slot.is_file():
        return json.loads(slot.read_text(encoding="utf-8")).get("winner")
    prompt = PROMPT.format(
        repo=tb.repo_block(case),
        title_a=a.title,
        abstract_a=a.abstract[:1500],
        title_b=b.title,
        abstract_b=b.abstract[:1500],
    )
    cfg = SimpleNamespace(provider="claude", claude_model=MODEL, timeout=90)
    rec: dict = {}
    for attempt in range(3):
        try:
            raw = complete(prompt, cfg, max_tokens=300)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"no JSON in: {raw[:100]}")
            winner = str(json.loads(match.group(0))["winner"]).strip().upper()
            if winner not in ("A", "B"):
                raise ValueError(f"winner {winner!r}")
            rec = {"winner": winner}
            break
        except (LLMError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            rec = {"error": f"{type(exc).__name__}: {exc}"}
            time.sleep(2 * (attempt + 1))
    if "winner" in rec:  # cache only successes — a cached failure would be permanent
        slot.parent.mkdir(parents=True, exist_ok=True)
        slot.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec.get("winner")


def bradley_terry(items: list[str], wins: dict[tuple[str, str], float]) -> dict[str, float]:
    """Zermelo MLE for BT log-strengths; ties entered as half-wins upstream."""
    strength = dict.fromkeys(items, 1.0)
    total_wins = {i: sum(w for (a, _), w in wins.items() if a == i) for i in items}
    pair_count: dict[tuple[str, str], float] = {}
    for (a, b), w in wins.items():
        key = (a, b) if a < b else (b, a)
        pair_count[key] = pair_count.get(key, 0.0) + w
    for _ in range(200):
        new = {}
        for i in items:
            denom = 0.0
            for j in items:
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                n = pair_count.get(key, 0.0)
                if n > 0:
                    denom += n / (strength[i] + strength[j])
            new[i] = (total_wins[i] + 1e-6) / denom if denom > 0 else strength[i]
        norm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {k: v / norm for k, v in new.items()}
        delta = max(abs(new[k] - strength[k]) for k in items)
        strength = new
        if delta < 1e-8:
            break
    return {k: math.log(v) for k, v in strength.items()}


def run_case(
    case: str,
    papers: list[tb.Paper],
    cache_dir: Path,
    with_anchors: bool,
    paraphrase: bool = False,
) -> dict:
    """All-pairs both orders (sampled above MAX_UNORDERED_PAIRS) + optional anchors."""
    papers = [p for p in papers if p.abstract]
    if len(papers) < 2 and not with_anchors:
        return {"n": len(papers), "skipped": "fewer than 2 papers"}
    pairs = list(itertools.combinations(range(len(papers)), 2))
    sampled = False
    if len(pairs) > MAX_UNORDERED_PAIRS:
        random.Random(case).shuffle(pairs)
        pairs = pairs[:MAX_UNORDERED_PAIRS]
        sampled = True
        print(f"    ! {case}: sampled {MAX_UNORDERED_PAIRS} of the full round-robin")
    anchors = anchor_papers(case, paraphrase) if with_anchors else []

    # Warm the cache concurrently; the serial aggregation below then hits cache only.
    from concurrent.futures import ThreadPoolExecutor

    ordered: list[tuple[tb.Paper, tb.Paper]] = []
    for i, j in pairs:
        ordered.append((papers[i], papers[j]))
        ordered.append((papers[j], papers[i]))
    for anc in anchors:
        for p in papers:
            ordered.append((anc, p))
            ordered.append((p, anc))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda t: one_comparison(case, t[0], t[1], cache_dir), ordered))

    wins: dict[tuple[str, str], float] = {}
    agree = disagree = 0
    for i, j in pairs:
        a, b = papers[i], papers[j]
        w1 = one_comparison(case, a, b, cache_dir)  # a shown first
        w2 = one_comparison(case, b, a, cache_dir)  # b shown first
        first = a.id if w1 == "A" else b.id if w1 == "B" else None
        second = b.id if w2 == "A" else a.id if w2 == "B" else None
        if first and second and first == second:
            agree += 1
            wins[(first, a.id if first == b.id else b.id)] = (
                wins.get((first, a.id if first == b.id else b.id), 0.0) + 2.0
            )
        elif first and second:
            disagree += 1
            wins[(a.id, b.id)] = wins.get((a.id, b.id), 0.0) + 1.0
            wins[(b.id, a.id)] = wins.get((b.id, a.id), 0.0) + 1.0
    for anc in anchors:
        for p in papers:
            w1 = one_comparison(case, anc, p, cache_dir)
            w2 = one_comparison(case, p, anc, cache_dir)
            first = anc.id if w1 == "A" else p.id if w1 == "B" else None
            second = p.id if w2 == "A" else anc.id if w2 == "B" else None
            if first and second and first == second:
                loser = p.id if first == anc.id else anc.id
                wins[(first, loser)] = wins.get((first, loser), 0.0) + 2.0
            elif first and second:
                wins[(anc.id, p.id)] = wins.get((anc.id, p.id), 0.0) + 1.0
                wins[(p.id, anc.id)] = wins.get((p.id, anc.id), 0.0) + 1.0
    items = [p.id for p in papers] + [a.id for a in anchors]
    strengths = bradley_terry(items, wins) if wins else {}
    result = {
        "n": len(papers),
        "pairs_compared": len(pairs),
        "sampled": sampled,
        "swap_agree": agree,
        "swap_disagree": disagree,
        "strengths": strengths,
    }
    if with_anchors and strengths:
        s_border = strengths.get("anchor_borderline")
        if s_border is not None:
            result["p_beats_borderline"] = {
                p.id: 1.0 / (1.0 + math.exp(-(strengths[p.id] - s_border)))
                for p in papers
                if p.id in strengths
            }
    return result


def testbed_bands(bed: str) -> dict[str, tb.CaseBand]:
    if bed == "a":
        return tb.load_testbed_a()
    if bed == "b":
        return tb.load_testbed_b()
    raise SystemExit(f"unknown testbed {bed!r} (E4 pre-registered A and B only)")


def summarize(bed: str, bands: dict[str, tb.CaseBand], results: dict[str, dict]) -> dict:
    strengths = {c: r.get("strengths", {}) for c, r in results.items()}
    summary: dict = {"testbed": bed, "model": MODEL}
    agree = sum(r.get("swap_agree", 0) for r in results.values())
    disagree = sum(r.get("swap_disagree", 0) for r in results.values())
    if agree + disagree:
        summary["swap_inconsistency"] = round(disagree / (agree + disagree), 3)
    summary["auc_band_judge2"] = tb.pooled_band_auc(bands, strengths, target=2)
    summary["auc_band_judge3"] = tb.pooled_band_auc(bands, strengths, target=3)
    p_beat = {c: r.get("p_beats_borderline", {}) for c, r in results.items()}
    if any(p_beat.values()):
        baselines = tb.baseline_nets(bands)
        per_case = {}
        for case, band in bands.items():
            policy = tb.policy_net(band, p_beat.get(case, {}))
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
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testbed", nargs="+", default=["a"], choices=["a", "b"])
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--paraphrase-check", action="store_true", help="borderline paraphrase arm")
    ap.add_argument("--case", help="single case, for a smoke run")
    args = ap.parse_args()

    tb.load_env()
    for bed in args.testbed:
        bands = testbed_bands(bed)
        if args.case:
            bands = {args.case: bands[args.case]}
        arm = "paraphrase" if args.paraphrase_check else "main"
        cache_dir = tb.EXP / "cache" / "pairwise" / bed / arm
        results: dict[str, dict] = {}
        for case in sorted(bands):
            t0 = time.time()
            results[case] = run_case(
                case,
                bands[case].admitted,
                cache_dir,
                with_anchors=not args.no_anchors,
                paraphrase=args.paraphrase_check,
            )
            r = results[case]
            if "skipped" not in r:
                print(
                    f"[{bed}:{case:10}] n={r['n']:2d} pairs={r['pairs_compared']:3d} "
                    f"agree={r['swap_agree']} disagree={r['swap_disagree']} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        summary = summarize(bed, bands, results)
        suffix = "_paraphrase" if args.paraphrase_check else ""
        out = tb.EXP / f"pairwise_{bed}{suffix}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8"
        )
        print(f"\n=== E4 {bed}{suffix} ===")
        for k, v in summary.items():
            if k not in ("per_case", "testbed", "model"):
                print(f"  {k}: {v}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
