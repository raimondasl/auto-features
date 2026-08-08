"""E1 — shuffled subset selection with a vote-share threshold.

The research pass's first-ranked experiment (evals/RESEARCH-score2-ranking.md §4.1).
Hypothesis: a single comparative *selection* call over a repo's whole admitted set,
repeated over shuffles, separates diffusion-like bands from vectordb-like bands where
pointwise scores cannot — and the "possibly none" option abstains on all-bad bands.

Per case: one prompt containing the prose-300 repo description plus every admitted paper
(title + abstract), instruction to select only the papers maintainers should actually act
on — possibly none — naming the component and the concrete change for each selection
(the evidence-first rubric, +7-11pp for Claude judges in the literature). R shuffles of
paper order at the API's default temperature; a paper's score is its selection share.

Pre-registered policy (fixed before any result was seen): digest = gate-3 papers + band-2
papers with share >= 2/3. Success: pooled within-band AUC >= 0.65 AND policy mean net@2
beats show-all with >= 14/22 per-case deltas >= 0 AND diffusion keeps its +10 AND at
least two of {numerics, compiler, vectordb, linter} improve. Kill: shares saturate
(mean > 0.8, sd < 0.1); empty selections never occur on linter/compiler; or AUC <= 0.55.

    uv run python evals/exp_select.py --model claude-sonnet-5 --testbed a
    uv run python evals/exp_select.py --model claude-haiku-4-5 --testbed a a300 b c

Every (case, shuffle) call is cached under .work/exp/cache/select/, so interrupted runs
resume for free and re-analysis costs nothing.
"""

from __future__ import annotations

import argparse
import json
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

PROMPT = """\
You are a senior research engineer reviewing candidate papers for one specific software
repository. Select ONLY the papers whose method the maintainers of THIS repository should
actually act on: a concrete, implementable improvement to this codebase. Topical relevance
is not enough — every candidate below is already on-topic. The bar is: would a competent
maintainer, after reading the paper, start a branch?

It is entirely possible that NONE of the candidates meet the bar. Selecting nothing is a
correct and common answer. Do not select a fixed fraction; judge each paper on its own.

# Repository
{repo}
# Candidate papers
{papers}
# Instructions
For each paper you select you MUST name (a) the specific repository component it improves
and (b) the concrete change a maintainer would make. If you cannot name both, do not
select the paper.

Respond with ONLY a JSON object, no other text:
{{"selected": [{{"id": "<paper id>", "component": "<component>", "change": "<change>"}}]}}
An empty list is a valid answer: {{"selected": []}}"""


def build_prompt(case: str, papers: list[tb.Paper], order: list[int]) -> tuple[str, list[str]]:
    """The selection prompt with papers in *order*; returns (prompt, label->paper.id map).

    Labels are positional (P01, P02...) so they carry no identity across shuffles — the
    mapping back to arXiv ids lives here, never in the prompt.
    """
    lines = []
    ids = []
    for slot, idx in enumerate(order, start=1):
        p = papers[idx]
        ids.append(p.id)
        lines.append(f"[P{slot:02d}] {p.title}\n{p.abstract[:1500]}\n")
    return (
        PROMPT.format(repo=tb.repo_block(case), papers="\n".join(lines)),
        ids,
    )


def parse_selected(raw: str, n: int) -> list[int]:
    """1-based slot numbers selected, from the model's JSON. Raises on malformed output."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:120]}")
    data = json.loads(match.group(0))
    if "selected" not in data or not isinstance(data["selected"], list):
        raise ValueError(f"no 'selected' list: {raw[:120]}")
    slots = []
    for item in data["selected"]:
        label = str(item["id"] if isinstance(item, dict) else item)
        m = re.search(r"(\d+)", label)
        if not m:
            raise ValueError(f"unparseable selection id {label!r}")
        slot = int(m.group(1))
        if not 1 <= slot <= n:
            raise ValueError(f"selection {label!r} out of range 1..{n}")
        slots.append(slot)
    return sorted(set(slots))


def testbed_bands(name: str) -> dict[str, tb.CaseBand]:
    if name == "a":
        return tb.load_testbed_a()
    if name == "a300":
        return tb.load_testbed_a300()
    if name == "b":
        return tb.load_testbed_b()
    if name == "c":
        # gate_full_pool's judged admits, grouped per case: tiny sets, pooled AUC only.
        bands: dict[str, tb.CaseBand] = {}
        for p in tb.load_testbed_c()["gate_full_pool"]:
            if p.gate >= 2:
                bands.setdefault(p.case, tb.CaseBand(case=p.case)).papers.append(p)
        return bands
    raise SystemExit(f"unknown testbed {name!r}")


def run_case(
    case: str,
    band: tb.CaseBand,
    model: str,
    shuffles: int,
    cache_dir: Path,
) -> dict:
    """Selection shares for one case's admitted set. Cached per (case, shuffle)."""
    papers = [p for p in band.admitted if p.abstract]
    dropped = len(band.admitted) - len(papers)
    if not papers:
        return {"n": 0, "dropped": dropped, "shares": {}, "empty_rate": None, "failures": 0}
    cfg = SimpleNamespace(provider="claude", claude_model=model, timeout=120)
    counts = {p.id: 0 for p in papers}
    valid = 0
    failures = 0
    empties = 0

    def one_shuffle(i: int) -> dict:
        """One (case, shuffle) call. Cached ONLY on success — a cached failure would
        otherwise become permanent and silently shrink the vote denominator forever."""
        slot = cache_dir / f"{case}_{i:02d}.json"
        if slot.is_file():
            return json.loads(slot.read_text(encoding="utf-8"))
        order = list(range(len(papers)))
        random.Random(f"{case}:{i}").shuffle(order)
        prompt, ids = build_prompt(case, papers, order)
        rec: dict = {"ids": ids, "raw": None, "selected": None}
        for attempt in range(3):
            try:
                # 4000: a full-set selection over 10 papers with per-paper component/change
                # fields overflowed 2000 and the truncation was CORRELATED with the verdict
                # (select-everything responses are the longest) — the max_tokens=500 judge
                # bug arriving by a different door.
                raw = complete(prompt, cfg, max_tokens=4000)
                rec["raw"] = raw
                rec["selected"] = parse_selected(raw, len(papers))
                break
            except (LLMError, ValueError, json.JSONDecodeError, KeyError) as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
        if rec["selected"] is not None:
            slot.parent.mkdir(parents=True, exist_ok=True)
            slot.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return rec

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as pool:
        recs = list(pool.map(one_shuffle, range(shuffles)))
    for rec in recs:
        if rec.get("selected") is None:
            failures += 1
            continue
        valid += 1
        if not rec["selected"]:
            empties += 1
        for s in rec["selected"]:
            counts[rec["ids"][s - 1]] += 1
    shares = {pid: (c / valid if valid else None) for pid, c in counts.items()}
    return {
        "n": len(papers),
        "dropped": dropped,
        "valid_shuffles": valid,
        "failures": failures,
        "empty_rate": (empties / valid) if valid else None,
        "shares": shares,
    }


def summarize(bands: dict[str, tb.CaseBand], results: dict[str, dict], testbed: str) -> dict:
    shares = {
        c: {k: v for k, v in r["shares"].items() if v is not None} for c, r in results.items()
    }
    summary: dict = {"testbed": testbed}

    flat = [v for per in shares.values() for v in per.values()]
    if flat:
        mean = sum(flat) / len(flat)
        sd = (sum((x - mean) ** 2 for x in flat) / len(flat)) ** 0.5
        summary["share_mean"] = round(mean, 3)
        summary["share_sd"] = round(sd, 3)

    summary["auc_band_judge2"] = tb.pooled_band_auc(bands, shares, target=2)
    summary["auc_band_judge3"] = tb.pooled_band_auc(bands, shares, target=3)

    # Pre-registered policy vs the same-file baselines, on band-shaped testbeds.
    baselines = tb.baseline_nets(bands)
    per_case = {}
    for case, band in bands.items():
        prob = shares.get(case, {})
        policy = tb.policy_net(band, prob)
        per_case[case] = {
            "policy": policy,
            "show_all": baselines[case]["show_all"],
            "delta": policy - baselines[case]["show_all"],
            "empty_rate": results.get(case, {}).get("empty_rate"),
        }
    summary["per_case"] = per_case
    deltas = [v["delta"] for v in per_case.values()]
    summary["policy_mean"] = round(sum(v["policy"] for v in per_case.values()) / len(per_case), 3)
    summary["show_all_mean"] = round(
        sum(v["show_all"] for v in per_case.values()) / len(per_case), 3
    )
    summary["deltas_nonneg"] = sum(1 for d in deltas if d >= 0)
    summary["sign_test"] = tb.sign_test(deltas)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--testbed", nargs="+", default=["a"], choices=["a", "a300", "b", "c"])
    ap.add_argument("--shuffles", type=int, default=15)
    ap.add_argument("--case", help="single case, for a smoke run")
    args = ap.parse_args()

    tb.load_env()
    tag = args.model.replace("claude-", "").replace(".", "")
    for bed in args.testbed:
        bands = testbed_bands(bed)
        if args.case:
            bands = {args.case: bands[args.case]}
        cache_dir = tb.EXP / "cache" / "select" / tag / bed
        results: dict[str, dict] = {}
        for case in sorted(bands):
            t0 = time.time()
            results[case] = run_case(case, bands[case], args.model, args.shuffles, cache_dir)
            r = results[case]
            if r["n"]:
                # empty_rate is None when every shuffle failed — print it, don't crash on it
                empty = f"{r['empty_rate']:.2f}" if r["empty_rate"] is not None else "n/a"
                print(
                    f"[{bed}:{case:10}] n={r['n']:2d} valid={r['valid_shuffles']:2d} "
                    f"empty={empty} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        summary = summarize(bands, results, bed)
        out = tb.EXP / f"select_{tag}_{bed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8"
        )
        print(f"\n=== E1 {bed} ({args.model}) ===")
        print(
            f"AUC(band, judge>=2)={summary['auc_band_judge2']:.3f}  "
            f"AUC(band, judge==3)={summary['auc_band_judge3']:.3f}"
        )
        print(
            f"policy mean net@2={summary['policy_mean']:+.2f} vs show-all "
            f"{summary['show_all_mean']:+.2f}  deltas>=0: {summary['deltas_nonneg']}"
            f"/{len(summary['per_case'])}  sign test p={summary['sign_test']['p']:.3f}"
        )
        print(f"share mean={summary.get('share_mean')} sd={summary.get('share_sd')}")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
