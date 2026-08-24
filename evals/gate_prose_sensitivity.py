"""How much does the gate's behaviour depend on what it is told the repository IS? ($0.)

    uv run python evals/gate_prose_sensitivity.py

§44 found two things about the actionability gate that were not what the arm was measuring.

* `systems` (redis) was handed *"Redis is a popular choice for developers worldwide due to its
  combination of speed, flexibility, and rich feature set"* and its gate went to **15 of 15
  actionable** — it stopped discriminating. §43.3 predicted that from stage 1 before any label was
  bought: marketing copy is agreeable about everything, and so is a gate reading it.
* `bio-align` was handed *"Minimap2 is a versatile sequence alignment program"* instead of a
  phishing warning, and the gate **dropped minimap2's own paper** — a self-referential paper it had
  been admitting while it had no idea what the project was.

Both say the gate's discrimination depends on the description it is given, and nobody has measured
how much. This does, **at $0 and with no new rule to overfit**, because the measurement was already
paid for: `evals/diagnose_triage.py` ran the shipped gate over the same 602 labelled papers under
six different repository descriptions, and every run is on disk with its judge label beside it.

**Two axes, and conflating them is the mistake this exists to avoid.**

* **Permissiveness** — what fraction of papers the gate admits. §44's `systems` observation is
  about this and nothing else.
* **Discrimination** — whether the papers it admits are the actionable ones, scored against the
  judge. A gate can become stricter without becoming better, and better without becoming stricter.

Reported per condition, over one fixed population, so the conditions are comparable to each other
rather than to a remembered number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

WORK = EVALS / ".work"
ACTIONABLE = 2

# The repository description each run gave the gate, poorest first. Ordering is by how much the
# project knows about the repo, NOT by any outcome — the point is to read the trend, and a trend
# ordered by its own result is a ranking, not a finding.
CONDITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "keywords",
        "diag_triage_keywords.json",
        "no prose at all — `prose_chars: 0`, the privacy setting",
    ),
    ("tagline", "diag_triage_tagline.json", "the packaging one-liner, 23-230 chars"),
    ("prose300", "diag_triage_prose300.json", "README prefix, 300 chars — WHAT SHIPS"),
    ("prose6000", "diag_triage_prose6000.json", "README prefix, 6000 chars"),
)


def load(name: str) -> dict[tuple[str, str], dict[str, int]]:
    path = WORK / name
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(r["case"], r["id"]): r for r in rows}


def auc(pairs: list[tuple[int, int]]) -> float | None:
    """Probability a random actionable paper outranks a random non-actionable one, on gate score.

    Ties count a half, which matters more here than anywhere else in this project: the gate is
    near-binary (§8) and most pairs ARE ties. An AUC that ignored them would flatter every
    condition equally and hide the thing being measured.
    """
    pos = [g for g, j in pairs if j >= ACTIONABLE]
    neg = [g for g, j in pairs if j < ACTIONABLE]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarise(rows: list[dict[str, int]]) -> dict[str, Any]:
    admitted = [r for r in rows if r["triage"] >= ACTIONABLE]
    act_admitted = sum(1 for r in admitted if r["judge"] >= ACTIONABLE)
    actionable = [r for r in rows if r["judge"] >= ACTIONABLE]
    recalled = sum(1 for r in actionable if r["triage"] >= ACTIONABLE)
    return {
        "n": len(rows),
        "admits": len(admitted) / len(rows) if rows else 0.0,
        "precision": act_admitted / len(admitted) if admitted else 0.0,
        "recall": recalled / len(actionable) if actionable else 0.0,
        # net@2 of showing exactly what the gate admits: the metric the product optimises.
        "net2": sum(1 if r["judge"] >= ACTIONABLE else -2 for r in admitted),
        "auc": auc([(r["triage"], r["judge"]) for r in rows]),
        "score_mix": {s: sum(1 for r in rows if r["triage"] == s) for s in range(4)},
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    loaded = {name: load(f) for name, f, _ in CONDITIONS}
    missing = [n for n, rows in loaded.items() if not rows]
    if missing:
        print(f"missing runs: {', '.join(missing)} — cannot compare conditions")
        return 1

    # One fixed population: papers every condition scored. Comparing conditions over different
    # paper sets would measure the sets.
    shared = set.intersection(*(set(rows) for rows in loaded.values()))
    print(f"population: {len(shared)} papers scored under all {len(CONDITIONS)} conditions")
    base = [loaded["prose300"][k] for k in shared]
    print(
        f"  of which the judge calls actionable: {sum(1 for r in base if r['judge'] >= ACTIONABLE)}"
    )

    print("\n" + "=" * 100)
    print("WHAT THE GATE DOES, BY WHAT IT WAS TOLD THE REPOSITORY IS")
    print("=" * 100)
    header = f"  {'condition':11} {'admits':>7} {'prec':>6} {'recall':>7} {'net@2':>7} {'AUC':>6}"
    print(f"{header}   description")
    stats = {}
    for name, _f, note in CONDITIONS:
        s = summarise([loaded[name][k] for k in shared])
        stats[name] = s
        a = "  n/a" if s["auc"] is None else f"{s['auc']:.3f}"
        print(
            f"  {name:11} {s['admits']:7.1%} {s['precision']:6.3f} {s['recall']:7.1%} "
            f"{s['net2']:+7d} {a:>6}   {note}"
        )

    print("\n  gate score distribution (the gate is near-binary, §8 — watch where the mass sits):")
    for name, _f, _n in CONDITIONS:
        print(f"    {name:11} {stats[name]['score_mix']}")

    poorest, shipped = stats["keywords"], stats["prose300"]
    print("\n" + "=" * 100)
    print("THE QUESTION §44 RAISED — does a poorer description make the gate MORE permissive?")
    print("=" * 100)
    delta = poorest["admits"] - shipped["admits"]
    print(f"  no prose at all admits {poorest['admits']:.1%} of the pool")
    print(f"  the shipped 300 chars  admits {shipped['admits']:.1%}")
    print(f"  difference: {delta:+.1%}")
    print(
        "\n  §44 saw one repository handed vague prose admit 15 of 15. That is a single case and\n"
        "  this is 4 conditions over one fixed population — it can say whether the EFFECT exists\n"
        "  in general, and it cannot say anything about vague-versus-absent, because no run here\n"
        "  gave the gate a description that was present but uninformative."
    )
    print(
        "\n  PERMISSIVENESS AND DISCRIMINATION ARE DIFFERENT AXES. Read `admits` against `AUC`:\n"
        "  a condition that admits less and separates no better has become strict, not smarter."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
