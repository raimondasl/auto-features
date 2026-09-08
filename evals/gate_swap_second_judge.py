"""Score NR-63's two gate arms under the second judge as well as the first.

NR-63 measured a Haiku gate against a GPT-5.6 Luna gate over one frozen pool, under GPT-5.5
alone, and its own limitations section names why that is not enough: NR-59 established that the
two judges order alike and threshold two orders of magnitude apart, and NR-63's only interval
excluding zero is a *threshold* result at `min>=3`. A threshold result is exactly the kind that
can move when the judge's severity changes.

Three labels, the same three `rung1_second_judge.py` registered, reported together:

    gpt           gpt >= 2                 what NR-63 published
    consensus     gpt >= 2 and son >= 1    does the gate difference survive both judges?
    sonnet_only   son >= 2                 what is it if the judge is simply swapped?

`sonnet_only` carries no bar, for the reason rung 1 gives: a threshold on its *level* would
measure the judge's severity rather than the system.

Deliberately NOT an extension of `rung1_second_judge.py`. That script's `report()` writes an
artifact bound to NR-52's own pre-registration, and widening its arm loaders would let one
registered comparison quietly answer for a different one. The helpers are imported — the
byte-identical rubric via `second_verdict`, the clone-drift guard via `verify_contexts` — so
there is exactly one implementation of each thing that matters.

    uv run python evals/gate_swap_second_judge.py --plan     # what it would cost
    uv run python evals/gate_swap_second_judge.py --judge    # buy the missing verdicts
    uv run python evals/gate_swap_second_judge.py            # report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

RES = EVALS / "results"
WORK = EVALS / ".work"
POOL = WORK / "pool-cut100"
OUT = WORK / "gate_swap_second_judge.json"

ARM_A = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T063132Z.json"  # haiku
ARM_B = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T074452Z.json"  # luna
GATE_A, GATE_B = "claude-haiku-4-5", "gpt-5.6-luna"
DEFAULT_MODEL = "claude-sonnet-5"


def arm(name: str) -> dict[str, list[dict[str, Any]]]:
    """Shown papers per case, as the arm returned them."""
    run = json.loads((RES / name).read_text(encoding="utf-8"))
    return {e["case"]: list(e["returned"]["reporadar_toppicks"]) for e in run}


def pool_meta() -> dict[tuple[str, str], dict[str, Any]]:
    """Title and abstract per (case, paper), from the frozen pool the first judge scored.

    The metadata has to come from where the FIRST judge got it. Two judges scoring two
    different texts is not a judge comparison, and the gold cache stores only scores — so a
    paper whose text cannot be rebuilt from the pool is dropped rather than judged against a
    reconstruction.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for f in POOL.glob("*.json"):
        case = f.stem
        for p in json.loads(f.read_text(encoding="utf-8")).get("candidates", []):
            pid = str(p.get("arxiv_id") or "")
            if pid:
                out[(case, pid)] = p
    return out


def paired(d: list[float]) -> dict[str, Any]:
    m, sd = st.mean(d), (st.stdev(d) if len(d) > 1 else 0.0)
    se = sd / math.sqrt(len(d)) if d else 0.0
    random.seed(7)
    obs, T = abs(m), 20000
    hits = 0
    for _ in range(T):
        flipped = st.mean([x if random.random() < 0.5 else -x for x in d])
        hits += abs(flipped) >= obs
    p = hits / T
    return {
        "delta": round(m, 3),
        "ci95": [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)],
        "p": round(p, 4),
        "w": sum(1 for x in d if x > 0),
        "l": sum(1 for x in d if x < 0),
        "t": sum(1 for x in d if x == 0),
    }


def _tag(shown: list[dict[str, Any]], case: str) -> list[dict[str, Any]]:
    for p in shown:
        p["_case"] = case
    return shown


def _load() -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    from second_judge import verify_contexts

    a, b = arm(ARM_A), arm(ARM_B)
    cases = sorted(set(a) & set(b))
    contexts, drifted = verify_contexts(cases)
    for c in cases:
        _tag(a.get(c, []), c)
        _tag(b.get(c, []), c)
    return a, b, sorted(contexts), drifted


def plan() -> int:
    from rung1_second_judge import cached_sonnet
    from second_judge import safe_paper_id

    a, b, cases, drifted = _load()
    cache, meta = cached_sonnet(DEFAULT_MODEL), pool_meta()
    print(f"cases in both arms: {len(cases)}")
    if drifted:
        print(f"DRIFTED (excluded): {sorted(drifted)}")
    need = missing_meta = 0
    for shown_by_case, name in ((a, GATE_A), (b, GATE_B)):
        tot = got = nom = 0
        for c in cases:
            for p in shown_by_case.get(c, []):
                pid = str(p.get("arxiv_id") or "")
                tot += 1
                if (c, pid) not in meta:
                    nom += 1
                elif cache.get((c, safe_paper_id(pid))) is not None:
                    got += 1
        need += tot - got - nom
        missing_meta += nom
        print(
            f"  {name:<18} {tot:>4} shown, {got:>4} cached, "
            f"{nom:>3} not in pool, need {tot - got - nom}"
        )
    print(f"\nfresh Sonnet verdicts required: {need}  (~${need * 0.012:.2f}-{need * 0.025:.2f})")
    if missing_meta:
        print(
            f"{missing_meta} shown paper(s) are not in the frozen pool "
            "and will be dropped, not judged."
        )
    return 0


def judge() -> int:
    from run_judge_eval import load_dotenv
    from second_judge import second_verdict, verify_contexts

    load_dotenv(EVALS / ".env")
    a, b, cases, _ = _load()
    contexts, _ = verify_contexts(cases)
    meta = pool_meta()
    bought = skipped = 0
    seen: set[tuple[str, str]] = set()
    for shown_by_case in (a, b):
        for c in cases:
            for p in shown_by_case.get(c, []):
                pid = str(p.get("arxiv_id") or "")
                if not pid or (c, pid) in seen:
                    continue
                seen.add((c, pid))
                paper = meta.get((c, pid))
                if paper is None:
                    skipped += 1
                    continue
                try:
                    second_verdict(c, contexts[c], {**paper, "arxiv_id": pid}, DEFAULT_MODEL)
                    bought += 1
                except Exception as exc:  # noqa: BLE001 — one bad paper must not lose the rest
                    print(f"  ! {c}/{pid}: {type(exc).__name__}: {str(exc)[:70]}")
                if bought % 25 == 0 and bought:
                    print(f"  [{bought}] verdicts", flush=True)
    print(f"\nbought/cached {bought} verdicts; {skipped} not in the pool and skipped")
    return 0


def report() -> int:
    from rung1_second_judge import cached_sonnet, label_consensus, label_gpt, label_sonnet
    from second_judge import safe_paper_id

    a, b, cases, drifted = _load()
    raw = cached_sonnet(DEFAULT_MODEL)

    def lookup(case: str, pid: str) -> int | None:
        return raw.get((case, safe_paper_id(pid)))

    def scored(shown: list[dict[str, Any]], label: Any) -> float:
        good = bad = 0
        for p in shown:
            g = p.get("judge_score")
            if g is None:
                continue
            v = label(int(g), lookup(p["_case"], str(p.get("arxiv_id") or "")))
            if v is None:
                continue
            good += 1 if v else 0
            bad += 0 if v else 1
        return good - 2 * bad

    out: dict[str, Any] = {
        "_comment": (
            "NR-63's two gate arms under both judges. Three labels registered by "
            "rung1_second_judge.py and reported together; sonnet_only carries no bar because a "
            "threshold on its level would measure the judge's severity rather than the system."
        ),
        "arms": {"A": {"gate": GATE_A, "file": ARM_A}, "B": {"gate": GATE_B, "file": ARM_B}},
        "n_cases": len(cases),
        "drifted_excluded": drifted,
        "labels": {},
    }
    for name, label in (
        ("gpt", label_gpt),
        ("consensus", label_consensus),
        ("sonnet_only", label_sonnet),
    ):
        va = [scored(a.get(c, []), label) for c in cases]
        vb = [scored(b.get(c, []), label) for c in cases]
        out["labels"][name] = {
            "mean": {GATE_A: round(st.mean(va), 3), GATE_B: round(st.mean(vb), 3)},
            "paired_luna_minus_haiku": paired([y - x for x, y in zip(va, vb, strict=True)]),
        }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["labels"], indent=1))
    print(f"\nwrote {OUT}")
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
