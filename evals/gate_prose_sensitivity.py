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

Both say the gate's discrimination depends on the description it is given, and nobody had measured
how much. This does, **at $0 and with no new rule to overfit**, because the measurement was already
paid for: `evals/diagnose_triage.py` ran the shipped gate over the same 602 labelled papers under
eight different repository descriptions, and every run is on disk with its judge label beside it.

**Two axes, and they push in OPPOSITE directions. Conflating them is the mistake this exists to
avoid, and the first version of this probe made it (§46).**

* **AMOUNT** — how much description the gate gets. Less amount, MORE admitted: 25.1% with no prose
  at all, 20.8% at the shipped 300 characters.
* **FIDELITY** — whether the words are the repository's own. Less fidelity, FEWER admitted: an LLM
  paraphrase admits 16.1% and its recall collapses to 52.4%, scoring *below no description at all*.

So "less information makes the gate permissive" is true of amount and false of fidelity, and §45.3
was wrong to record the fidelity question as unreachable — three runs on disk answer it.

**Permissiveness and discrimination are also different things**, and are reported separately: a
gate can become stricter without becoming better.
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

# (name, artifact, axis, description). Ordered within each axis by how much the gate is told,
# NOT by any outcome — a trend ordered by its own result is a ranking, not a finding.
CONDITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("keywords", "diag_triage_keywords.json", "amount", "no prose at all — `prose_chars: 0`"),
    ("tagline", "diag_triage_tagline.json", "amount", "the packaging one-liner, 23-230 chars"),
    ("prose300", "diag_triage_prose300.json", "amount", "README prefix, 300 chars — WHAT SHIPS"),
    ("prose6000", "diag_triage_prose6000.json", "amount", "README prefix, 6000 chars"),
    ("summary", "diag_triage_summary.json", "fidelity", "LLM PARAPHRASE — fluent, no term-of-art"),
    ("summary_nogaps", "diag_triage_summary_nogaps.json", "fidelity", "paraphrase, no gaps block"),
    ("summary_hedged", "diag_triage_summary_hedged.json", "fidelity", "paraphrase, hedged"),
    ("extractive", "diag_triage_extractive.json", "fidelity", "VERBATIM, semantically selected"),
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

    loaded = {name: load(f) for name, f, _a, _d in CONDITIONS}
    missing = [n for n, rows in loaded.items() if not rows]
    if missing:
        print(f"missing runs: {', '.join(missing)} — cannot compare conditions")
        return 1

    # One fixed population: papers every condition scored. Comparing conditions over different
    # paper sets would measure the sets.
    shared = set.intersection(*(set(rows) for rows in loaded.values()))
    base = [loaded["prose300"][k] for k in shared]
    print(f"population: {len(shared)} papers scored under all {len(CONDITIONS)} conditions")
    print(f"  judged actionable: {sum(1 for r in base if r['judge'] >= ACTIONABLE)}")

    stats = {n: summarise([loaded[n][k] for k in shared]) for n, _f, _a, _d in CONDITIONS}

    for axis, title in (
        ("amount", "AXIS 1 — HOW MUCH the gate is told. Less amount, MORE admitted."),
        ("fidelity", "AXIS 2 — WHOSE WORDS it is told in. Less fidelity, FEWER admitted."),
    ):
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        head = f"  {'condition':16} {'admits':>7} {'prec':>6} {'recall':>7} {'net@2':>7} {'AUC':>6}"
        print(f"{head}   description")
        for name, _f, cond_axis, note in CONDITIONS:
            if cond_axis != axis:
                continue
            s = stats[name]
            a = "  n/a" if s["auc"] is None else f"{s['auc']:.3f}"
            print(
                f"  {name:16} {s['admits']:7.1%} {s['precision']:6.3f} {s['recall']:7.1%} "
                f"{s['net2']:+7d} {a:>6}   {note}"
            )
        if axis == "fidelity":
            print(
                f"  {'prose300':16} {stats['prose300']['admits']:7.1%} "
                f"{stats['prose300']['precision']:6.3f} {stats['prose300']['recall']:7.1%} "
                f"{stats['prose300']['net2']:+7d} {stats['prose300']['auc']:6.3f}"
                "   verbatim, POSITIONALLY selected (carried over)"
            )

    none_, shipped = stats["keywords"], stats["prose300"]
    para, extract = stats["summary"], stats["extractive"]

    print("\n" + "=" * 100)
    print("WHAT THE TWO AXES SAY, TOGETHER")
    print("=" * 100)
    print(
        f"  AMOUNT   no prose {none_['admits']:.1%} admitted -> shipped {shipped['admits']:.1%}. "
        f"Knowing more, the gate REFUSES more,\n"
        f"           and refuses better: precision {none_['precision']:.3f} -> "
        f"{shipped['precision']:.3f}, AUC {none_['auc']:.3f} -> {shipped['auc']:.3f}.\n"
        f"           net@2 {none_['net2']:+d} -> {shipped['net2']:+d}, and recall FALLS "
        f"{none_['recall']:.1%} -> {shipped['recall']:.1%} — the gain is\n"
        f"           refusing better, not finding more."
    )
    print(
        f"\n  FIDELITY paraphrase {para['admits']:.1%} admitted, recall {para['recall']:.1%}, "
        f"net@2 {para['net2']:+d} — BELOW no prose at all ({none_['net2']:+d}).\n"
        f"           A description present but not in the repo's OWN words gives the gate\n"
        f"           nothing to match, and it refuses. Opposite direction to\n"
        f"           amount, and §45 predicted the wrong sign for it."
    )
    print(
        f"\n  SELECTION barely matters once the words are verbatim: extractive "
        f"{extract['net2']:+d} / AUC {extract['auc']:.3f}\n"
        f"           against the positional prefix {shipped['net2']:+d} / {shipped['auc']:.3f}. "
        f"WHOSE words beats WHICH words."
    )
    print(
        "\n  NOT ANSWERED: §44's `systems` case, which went to 15 of 15. Its prose was neither\n"
        "  absent nor paraphrased — it was BROAD, full of real systems vocabulary with no scope.\n"
        "  Neither axis here isolates that, and constructing prose to test it would mean writing\n"
        "  the treatment to get a result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
