"""Rung 1: is the +0.54 margin over Opus 5 a property of RepoRadar or of GPT-5.5? [NR-52]

Pre-registered in `evals/PREREG-rung1.md`, committed before any margin was computed under any
label. Read that first; this script only executes it.

Every direction on the ladder in `RESEARCH-net2-directions.md` operates where the judges agree
least — GPT-Sonnet kappa 0.199 on the band, and the 80 gate-withheld papers run 74% actionable
under GPT against 26% under a consensus label, so the *sign* of every rescue idea's expected
value flips with the judge. This buys the missing Sonnet verdicts and re-scores two arms that
were already paid for.

Three labels, all registered in advance, all reported:

    gpt           gpt >= 2                the published margin
    consensus     gpt >= 2 and son >= 1   does it survive when both judges must agree?
    sonnet_only   son >= 2                what is it if we just swap the judge?

Only the consensus margin carries a kill condition (sign flip, or |delta| > 0.5/case). A bar on
the Sonnet-only level would be measuring the judge's severity, not the system.

Routed through `second_judge.second_verdict` — byte-identical rubric, cached outside the gold
cache — and guarded by `second_judge.verify_contexts`, which excludes any case whose repo clone
drifted since its GPT verdicts were cached: there the stored label answers a question we can no
longer rebuild, so comparing judges would compare two prompts.

    uv run python evals/rung1_second_judge.py --plan     # $0: coverage, drift, cost
    uv run python evals/rung1_second_judge.py --judge    # ~$6-12 of Sonnet, resumable
    uv run python evals/rung1_second_judge.py            # $0: the three margins
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from band_testbeds import sign_test  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402
from second_judge import (  # noqa: E402
    DEFAULT_MODEL,
    safe_paper_id,
    verify_contexts,
)

from reporadar.paper_id import dedup_id  # noqa: E402

RES = EVALS / "results"
WORK = EVALS / ".work"
SHIP = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T034455Z.json"
SHIP_REPAIR = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260830T075622Z.json"
OPUS5 = EVALS / "gold_spread_v2_opus5.json"
POOL = WORK / "pool-cut100"
FROZEN = EVALS / "rung1_second_judge.json"

BAR = 0.5  # pre-registered: |consensus - gpt| must stay within this, sign preserved
BIG_LOSSES = ("mat-mlip", "mat-chgpot", "cv", "llminfer", "mat-toolkit", "numerics")


def shipped_arm() -> dict[str, list[dict[str, Any]]]:
    """Shown papers per case, from the arm the comparator claim rests on."""

    def load(name: str) -> dict[str, list[dict[str, Any]]]:
        run = json.loads((RES / name).read_text(encoding="utf-8"))
        return {e["case"]: list(e["returned"]["reporadar_toppicks"]) for e in run}

    out = load(SHIP)
    out["bio-mdtraj"] = load(SHIP_REPAIR)["bio-mdtraj"]
    return out


def opus5_arm() -> dict[str, list[dict[str, Any]]]:
    """Opus 5 draw 1, picks that carry a verdict.

    Hallucinated and unjudgeable picks are ABSENT rather than scored negative — the same
    void-not-null rule `freeze_opus5_arm.py` applies, which is what the comparator claim rests
    on and what this must reproduce exactly to be comparing the same arm.
    """
    rows = json.loads(OPUS5.read_text(encoding="utf-8"))["results"]
    out: dict[str, list[dict[str, Any]]] = {}
    for key, row in rows.items():
        draw, case = key.split("/", 1)
        if draw != "1" or row.get("status") != "ok":
            continue
        scores = row.get("scores") or {}
        out[case] = [
            {"arxiv_id": p, "judge_score": int(scores[p])}
            for p in (row.get("picks") or [])
            if p in scores
        ]
    return out


def pool_metadata(case: str) -> dict[str, dict[str, Any]]:
    f = POOL / f"{case}.json"
    if not f.is_file():
        return {}
    return {str(c["arxiv_id"]): c for c in json.loads(f.read_text(encoding="utf-8"))["candidates"]}


def cached_sonnet(model: str = DEFAULT_MODEL) -> dict[tuple[str, str], int]:
    root = WORK / "second_judge" / model
    out: dict[tuple[str, str], int] = {}
    for f in root.rglob("*.json"):
        try:
            out[(f.parent.name, f.stem)] = int(json.loads(f.read_text(encoding="utf-8"))["score"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            continue
    return out


def son_of(cache: dict[tuple[str, str], int], case: str, pid: Any) -> int | None:
    return cache.get((case, safe_paper_id(str(pid))))


def label_gpt(g: int, s: int | None) -> bool | None:
    return g >= 2


def label_consensus(g: int, s: int | None) -> bool | None:
    return None if s is None else (g >= 2 and s >= 1)


def label_sonnet(g: int, s: int | None) -> bool | None:
    return None if s is None else s >= 2


LABELS = {"gpt": label_gpt, "consensus": label_consensus, "sonnet_only": label_sonnet}


def net2(picks: list[dict[str, Any]], label: Any, cache: Any, case: str) -> tuple[int, int]:
    """(net@2, n_scored) under one label. A paper the label cannot evaluate is VOID.

    Void, never zero: a missing Sonnet verdict means unmeasured, and scoring it as
    non-actionable would charge -2 for our own coverage gap (C-4, C-30).
    """
    total = n = 0
    for p in picks:
        v = label(int(p["judge_score"]), son_of(cache, case, p["arxiv_id"]))
        if v is None:
            continue
        n += 1
        total += 1 if v else -2
    return total, n


def plan() -> int:
    ship, opus = shipped_arm(), opus5_arm()
    cache = cached_sonnet()
    cases = sorted(set(ship) & set(opus))
    contexts, drifted = verify_contexts(cases)
    print(f"cases in both arms: {len(cases)}")
    print(f"context-verified:   {len(contexts)}")
    if drifted:
        print(f"DRIFTED (excluded): {len(drifted)} -> {sorted(drifted)}")
    need = 0
    for arm, name in ((ship, "shipped"), (opus, "opus5")):
        tot = sum(len(arm.get(c, [])) for c in contexts)
        got = sum(
            1
            for c in contexts
            for p in arm.get(c, [])
            if son_of(cache, c, p["arxiv_id"]) is not None
        )
        need += tot - got
        print(f"  {name:<8} {tot:>4} papers, {got:>4} cached ({got / tot:.1%}), need {tot - got}")
    print(f"\nfresh Sonnet verdicts required: {need}  (~${need * 0.012:.0f}-{need * 0.025:.0f})")
    return 0


def judge() -> int:
    """Buy the missing verdicts. Resumable — `second_verdict` caches per paper.

    Metadata comes from where the FIRST judge got it: the frozen pool for shipped papers,
    `verify.resolve_references` for Opus 5's picks (the same call `gold_spread.judge_row`
    made). Two judges scoring two different texts is not a judge comparison, and the gold
    cache stores only scores, so the text has to be rebuilt the way it was built originally.
    """
    from run_judge_eval import load_dotenv
    from second_judge import second_verdict
    from verify import resolve_references

    load_dotenv(EVALS / ".env")
    ship, opus = shipped_arm(), opus5_arm()
    cases = sorted(set(ship) & set(opus))
    contexts, drifted = verify_contexts(cases)
    if drifted:
        print(f"EXCLUDED (context drift): {sorted(drifted)}")
    cache = cached_sonnet()
    bought = void = 0

    for n, case in enumerate(sorted(contexts), start=1):
        ctx = contexts[case]
        meta = pool_metadata(case)
        todo_ship = [p for p in ship.get(case, []) if son_of(cache, case, p["arxiv_id"]) is None]
        todo_opus = [p for p in opus.get(case, []) if son_of(cache, case, p["arxiv_id"]) is None]
        if not todo_ship and not todo_opus:
            continue
        papers: list[dict[str, Any]] = []
        for p in todo_ship:
            m = meta.get(str(p["arxiv_id"]))
            papers.append({**p, "abstract": (m or {}).get("abstract", "")})
        if todo_opus:
            try:
                resolved, *_ = resolve_references([str(p["arxiv_id"]) for p in todo_opus], [])
            except Exception as exc:  # noqa: BLE001 -- a resolver failure is void, not zero
                print(f"  ! {case}: resolver failed: {type(exc).__name__}: {str(exc)[:70]}")
                resolved = []
            # `dedup_id`, not `safe_paper_id`: the resolver hands back VERSIONED ids
            # (`2108.13264v4`) while a pick is unversioned (`2108.13264`), so matching on the
            # filename rule silently missed every arXiv paper and scored 229 of them void.
            # One normaliser, everywhere — C-14, and this is the tenth site to learn it.
            by_id = {dedup_id(str(r["arxiv_id"])): r for r in resolved}
            for p in todo_opus:
                r = by_id.get(dedup_id(str(p["arxiv_id"])))
                if r is None:
                    void += 1
                    continue
                papers.append({**r, "arxiv_id": p["arxiv_id"]})
        for paper in papers:
            if not str(paper.get("abstract") or "").strip():
                void += 1
                continue
            try:
                second_verdict(case, ctx, paper, DEFAULT_MODEL)
                bought += 1
            except Exception as exc:  # noqa: BLE001 -- one bad paper must not lose the rest
                void += 1
                print(f"  ! {case}/{paper['arxiv_id']}: {type(exc).__name__}: {str(exc)[:70]}")
        print(f"  [{n}/{len(contexts)}] {case:<16} bought {bought} so far, void {void}")
    print(f"\nbought {bought} verdicts; {void} void (unresolved or abstract-less)")
    return 0


def report() -> int:
    ship, opus = shipped_arm(), opus5_arm()
    both = sorted(set(ship) & set(opus))
    contexts, drifted = verify_contexts(both)
    cache = cached_sonnet()
    cases = sorted(contexts)

    out: dict[str, Any] = {
        "_comment": (
            "NR-52 / rung 1. Pre-registered in evals/PREREG-rung1.md and COMMITTED BEFORE any "
            "margin was computed under any label. Three labels registered together and all "
            "reported: gpt (the published margin), consensus (gpt>=2 AND sonnet>=1, the only "
            "one carrying a kill condition), sonnet_only (no bar -- a threshold on its level "
            "would measure the judge's severity rather than the system). Derived by "
            "evals/rung1_second_judge.py; pinned by tests/test_rung1_second_judge.py."
        ),
        "excluded_context_drift": sorted(drifted),
        "n_cases": len(cases),
        "labels": {},
    }

    for name, fn in LABELS.items():
        per_case = {}
        for c in cases:
            sn, s_n = net2(ship[c], fn, cache, c)
            on, o_n = net2(opus[c], fn, cache, c)
            per_case[c] = {"rr": sn, "opus5": on, "delta": sn - on, "rr_n": s_n, "opus5_n": o_n}
        d = [float(per_case[c]["delta"]) for c in cases]
        lo, hi = paired_bootstrap(d)
        sg = sign_test(d)
        out["labels"][name] = {
            "rr_mean_net2": round(st.mean(per_case[c]["rr"] for c in cases), 2),
            "opus5_mean_net2": round(st.mean(per_case[c]["opus5"] for c in cases), 2),
            "margin": round(st.mean(d), 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "wins": sg["pos"],
            "losses": sg["neg"],
            "ties": sg["ties"],
            "sign_p": round(sg["p"], 4),
            "rr_papers_scored": sum(per_case[c]["rr_n"] for c in cases),
            "opus5_papers_scored": sum(per_case[c]["opus5_n"] for c in cases),
            "per_case": per_case,
        }

    # ── How much does the consensus label actually BIND? ──────────────────────────────
    # The bar was set on a label whose power was never checked. `Sonnet >= 1` is a weak
    # constraint, so if Sonnet almost never scores 0 the consensus label reduces to GPT by
    # construction and the +-0.5 test passes whatever the truth is. Measured, not assumed.
    binding = {}
    for nm, arm in (("shipped", ship), ("opus5", opus)):
        g2 = [(c, p) for c in cases for p in arm[c] if int(p["judge_score"]) >= 2]
        demoted = sum(1 for c, p in g2 if (son_of(cache, c, p["arxiv_id"]) or 0) < 1)
        hist: dict[str, int] = {}
        for c in cases:
            for p in arm[c]:
                key = str(son_of(cache, c, p["arxiv_id"]))
                hist[key] = hist.get(key, 0) + 1
        act = sum(v2 for k2, v2 in hist.items() if k2 in ("2", "3"))
        tot = sum(hist.values())
        binding[nm] = {
            "gpt_actionable": len(g2),
            "demoted_by_consensus": demoted,
            "demotion_rate": round(demoted / len(g2), 4) if g2 else None,
            "sonnet_score_hist": hist,
            "sonnet_precision": round(act / tot, 4) if tot else None,
        }
    out["consensus_label_binding"] = {
        "_comment": (
            "The power check the pre-registration should have contained. `Sonnet >= 1` demotes "
            "under 2% of GPT-actionable papers in either arm, so the consensus label is very "
            "nearly GPT itself and the +-0.5 bar was close to unfalsifiable. Reported because a "
            "bar that passes almost regardless of the truth is a defect in the bar -- the same "
            "shape NR-49 recorded, where a null cleared one; here a pass is near-vacuous."
        ),
        **binding,
        "break_even_precision": round(2 / 3, 4),
    }

    g = out["labels"]["gpt"]["margin"]
    k = out["labels"]["consensus"]["margin"]
    big = {
        c: {
            "gpt": out["labels"]["gpt"]["per_case"][c]["delta"],
            "consensus": out["labels"]["consensus"]["per_case"][c]["delta"],
        }
        for c in BIG_LOSSES
        if c in cases
    }
    persisted = sum(1 for v in big.values() if v["gpt"] < 0 and v["consensus"] <= 0.5 * v["gpt"])
    out["pre_registered"] = {
        "bar": BAR,
        "criterion": (
            "consensus margin within +-0.5/case of gpt with sign preserved, AND >=4 of the 6 "
            "big science losses persisting at >=50% of their gpt magnitude"
        ),
        "committed_before_any_margin": True,
        "sonnet_only_has_no_bar": (
            "Its level is dominated by one judge's severity -- both arms are expected to go "
            "negative -- so a bar there would measure the judge, not the system. Its SIGN is "
            "the informative part."
        ),
        "scope_limit": (
            "The +-0.5 read is descriptive gating, not an equivalence test: the consensus "
            "margin's own SE is ~1.0/case at n=37. A pass means robust among the judges we "
            "have, never objective -- a cutoff both models share (NR-43) is undetectable by "
            "any two-judge construction."
        ),
    }
    out["verdict"] = {
        "gpt_margin": g,
        "consensus_margin": k,
        "sonnet_only_margin": out["labels"]["sonnet_only"]["margin"],
        "abs_shift": round(abs(k - g), 2),
        "sign_preserved": bool((k > 0) == (g > 0)),
        "within_bar": bool(abs(k - g) <= BAR),
        "big_losses_persisting": persisted,
        "big_losses": big,
        "passes": bool(abs(k - g) <= BAR and (k > 0) == (g > 0) and persisted >= 4),
        "but_the_bar_was_weak": (
            "The consensus label demotes 0.7% of shipped and 1.7% of Opus 5 GPT-actionable "
            "papers, so it is nearly GPT by construction and this pass carries little "
            "information. Recorded rather than repaired: moving a bar after seeing the data is "
            "the failure NR-49 documented, so the pass stands as written and its weakness is "
            "measured beside it."
        ),
        "sonnet_only_sign_flips": bool((out["labels"]["sonnet_only"]["margin"] > 0) != (g > 0)),
        "sonnet_only_reading": (
            "The pre-registration named the Sonnet-only SIGN as the informative part, and it "
            "flips. Under Sonnet, RepoRadar scores -2.03/case against Opus 5's +1.38: our shown "
            "papers run 58.5% actionable, BELOW net@2's 2/3 break-even, while Opus 5's run "
            "71.4%, above it. The prediction written in advance -- that a harsher judge would "
            "push BOTH arms negative and penalise the arm showing more -- is wrong in both "
            "halves, and the direction is against us."
        ),
        "gpt_margin_is_draw_dependent": (
            "This control (20260830T034455Z, RR mean +5.51) gives a GPT margin of +0.32; "
            "opus5_arm.json's control (20260827T213701Z, RR mean +5.73) gives +0.54 against an "
            "identical Opus 5 arm. Same config, different draw, 0.22/case apart -- C-7 applies "
            "to our own side too, and the headline number is not draw-stable."
        ),
    }
    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"{len(cases)} cases" + (f"; EXCLUDED {sorted(drifted)}" if drifted else "; no drift"))
    header = f"{'label':<14}{'RepoRadar':>11}{'Opus 5':>9}{'margin':>9}{'CI95':>19}{'w/l/t':>12}"
    print("\n" + header)
    for name in ("gpt", "consensus", "sonnet_only"):
        v = out["labels"][name]
        ci = f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
        wlt = f"{v['wins']}/{v['losses']}/{v['ties']}"
        print(
            f"{name:<14}{v['rr_mean_net2']:>+11.2f}{v['opus5_mean_net2']:>+9.2f}"
            f"{v['margin']:>+9.2f}{ci:>19}{wlt:>12}"
        )
    print("\npapers scored (void excluded, never zeroed):")
    for name in LABELS:
        v = out["labels"][name]
        print(
            f"  {name:<14} RR {v['rr_papers_scored']:>3}/306"
            f"   Opus5 {v['opus5_papers_scored']:>3}/357"
        )
    print(f"\nbig science losses persisting at >=50%: {persisted}/6")
    for c, v in big.items():
        print(f"  {c:<14} gpt {v['gpt']:>+4}  consensus {v['consensus']:>+4}")
    v = out["verdict"]
    print(
        f"\nPRE-REGISTERED BAR: |consensus - gpt| = {v['abs_shift']:.2f} (<= {BAR}? "
        f"{v['within_bar']}), sign preserved {v['sign_preserved']}, big losses {persisted}/6"
    )
    b = out["consensus_label_binding"]
    print(
        f"\nBAR POWER: consensus demotes {b['shipped']['demoted_by_consensus']}/"
        f"{b['shipped']['gpt_actionable']} shipped and {b['opus5']['demoted_by_consensus']}/"
        f"{b['opus5']['gpt_actionable']} opus5 GPT-actionable papers "
        f"({b['shipped']['demotion_rate']:.1%} / {b['opus5']['demotion_rate']:.1%}) "
        f"-- the label is nearly GPT itself"
    )
    print(
        f"SONNET PRECISION: shipped {b['shipped']['sonnet_precision']:.3f} vs "
        f"opus5 {b['opus5']['sonnet_precision']:.3f} (break-even {2 / 3:.3f})"
    )
    print("VERDICT: " + ("PASS as written" if v["passes"] else "KILL"))
    if v["sonnet_only_sign_flips"]:
        print("  BUT the Sonnet-only margin flips sign, which the prereg named as informative.")
    print(f"wrote {FROZEN.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    if args.plan:
        return plan()
    if args.judge:
        return judge()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
