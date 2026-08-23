"""§28 PRIMARY (H-B): does "proposes no new method" predict a dud among score-3 papers? (~$0.40)

    uv run python evals/proposes_method.py --dry-run   # population + cost, $0
    uv run python evals/proposes_method.py             # classify, then report
    uv run python evals/proposes_method.py --report    # re-derive from cache, $0

§26 refuted the tool-name mechanism. §29 found H-A ("the repo already has it") unsupported, and
found something better: being cited does not make a paper a dud — 7 of 10 cited score-3 papers
are judged actionable. H-B is the remaining candidate: **the misfires are papers that propose
nothing to have** — surveys, benchmarks, position papers. Maximal topical overlap, zero portable
method. Three of the ten misfires that reach a real user are exactly that shape.

**The prompt is fixed here, in the file, and it deliberately never mentions the repository.**
H-B is a paper-intrinsic property: whether an abstract proposes a method is true of the paper
whatever repo is asking. Excluding repo context makes the classification cheaper, and — far more
importantly — makes it a different question from the one the gate and the judges answer. That is
what the KILL clause below polices.

**BAR, amended before running and for a stated reason.** §28.6 set "≥ 20 percentage points".
§29.2 found that a points bar is not judge-comparable when the judges have different base rates,
and systematically favours the stricter one — Sonnet's base rate is roughly double GPT's, so the
same effect reads as 16.7 points or 44.7 depending on who scored it. The bar is therefore a
**ratio ≥ 2.0×, holding under both judges**, which at these base rates is the comparable
restatement of the original. Amended after seeing the *secondary's* data and before seeing the
primary's; recorded so the change is auditable rather than convenient.

**KILL** — if the classifier is re-judging actionability under another name, a positive result
means nothing. Checked two ways, both automatic: the classifier must not agree with the judge
more than it agrees with itself across judges, and its output must not be near-perfectly
predicted by the judge label. Ten papers are printed for hand-checking regardless.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from label_pool import fisher_exact  # noqa: E402
from score3_mechanism import collect  # noqa: E402
from second_judge import (  # noqa: E402
    ACTIONABLE,
    DEFAULT_MODEL,
    cohens_kappa,
    safe_paper_id,
    second_cache_path,
)
from second_judge import _load_env as load_env  # noqa: E402

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.llm_client import LLMError, complete  # noqa: E402

WORK = EVALS / ".work"
CLASSIFY_CACHE = WORK / "proposes_method"
RATIO_BAR = 2.0  # §29.2: ratio, not points, because base rates differ between the judges

# Fixed here so it cannot be tuned after the labels are joined. Note what is absent: the
# repository. This asks about the PAPER.
PROMPT = """\
You are classifying a research abstract by what kind of contribution it makes. This is not a
relevance judgement and no particular reader is implied.

Answer one question: does this abstract PROPOSE a new method, technique, model, algorithm or
system that someone could implement?

Answer "proposes" if the paper introduces something implementable, even incrementally.
Answer "no_proposal" if the paper instead surveys a field, benchmarks or compares existing
methods, reviews representations, or argues a position, without introducing something new to
implement.

Reply with JSON only: {"contribution": "proposes" | "no_proposal", "why": "<8 words>"}

Title: %(title)s

Abstract: %(abstract)s
"""


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def classify(row: dict[str, Any], model: str) -> str:
    path = CLASSIFY_CACHE / model / row["case"] / f"{safe_paper_id(row['arxiv_id'])}.json"
    if path.is_file():
        return str(json.loads(path.read_text(encoding="utf-8"))["contribution"])
    prompt = PROMPT % {"title": row["title"], "abstract": row["abstract"][:1800]}
    cfg = SuggestionsConfig(provider="claude", claude_model=model, timeout=90)
    raw = complete(prompt, cfg, max_tokens=300)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < 0:
        raise ValueError(f"no JSON in response: {raw[:120]}")
    data = json.loads(raw[a : b + 1])
    if data.get("contribution") not in ("proposes", "no_proposal"):
        raise ValueError(f"bad label: {data!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(data["contribution"])


def report_arm(label: str, rows: list[dict[str, Any]], outcome: str) -> dict[str, Any] | None:
    no_prop = [r for r in rows if r["contribution"] == "no_proposal"]
    prop = [r for r in rows if r["contribution"] == "proposes"]
    if not no_prop or not prop:
        print(f"  {label:22} one arm empty ({len(no_prop)} vs {len(prop)}) — not reported")
        return None
    ka, na = sum(1 for r in no_prop if r.get(outcome)), len(no_prop)
    kb, nb = sum(1 for r in prop if r.get(outcome)), len(prop)
    la, ha = wilson(ka, na)
    lb, hb = wilson(kb, nb)
    ratio = (ka / na) / (kb / nb) if kb else float("inf")
    p = fisher_exact(ka, na - ka, kb, nb - kb)
    print(
        f"  {label:22} no_proposal {ka:2d}/{na:2d} = {ka / na:.3f} [{la:.3f},{ha:.3f}]   "
        f"proposes {kb:2d}/{nb:2d} = {kb / nb:.3f} [{lb:.3f},{hb:.3f}]"
    )
    print(
        f"  {'':22}   gap {ka / na - kb / nb:+.3f} points   RATIO {ratio:.2f}x   p={p:.4f}"
        f"   (bar {RATIO_BAR:.1f}x)"
    )
    return {"ratio": ratio, "p": p, "no_proposal": [ka, na], "proposes": [kb, nb]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rows, _stats = collect()
    for r in rows:
        p = second_cache_path(DEFAULT_MODEL, r["case"], r["arxiv_id"])
        if p.is_file():
            r["sonnet_non_actionable"] = (
                int(json.loads(p.read_text(encoding="utf-8"))["score"]) < ACTIONABLE
            )
    cached = sum(
        1
        for r in rows
        if (
            CLASSIFY_CACHE / args.model / r["case"] / f"{safe_paper_id(r['arxiv_id'])}.json"
        ).is_file()
    )
    print(
        f"population: {len(rows)} score-3 papers   classified already: {cached}   "
        f"to call: {len(rows) - cached}"
    )
    if args.dry_run:
        print("\ndry run — nothing was called.")
        return 0

    if not args.report:
        load_env()
    failed = 0
    for i, r in enumerate(rows, 1):
        try:
            r["contribution"] = classify(r, args.model)
        except (LLMError, ValueError, KeyError) as exc:
            failed += 1
            print(f"  ! {r['case']}/{r['arxiv_id']}: {str(exc)[:80]}")
        if i % 25 == 0:
            print(f"  classified {i}/{len(rows)}", flush=True)
    rows = [r for r in rows if "contribution" in r]
    if failed:
        print(f"  ! {failed} unclassified and EXCLUDED, never defaulted")

    n_no = sum(1 for r in rows if r["contribution"] == "no_proposal")
    print(f"\n  classifier: {n_no}/{len(rows)} papers propose nothing ({n_no / len(rows):.0%})")

    print("\n" + "=" * 100)
    print("PRIMARY (H-B) — does 'proposes no new method' predict non-actionability?")
    print("=" * 100)
    out = {
        "gpt": report_arm("GPT-5.5", rows, "non_actionable"),
        "sonnet": report_arm(
            "Sonnet", [r for r in rows if "sonnet_non_actionable" in r], "sonnet_non_actionable"
        ),
    }

    print("\n" + "=" * 100)
    print("KILL CHECK — is the classifier just re-judging actionability?")
    print("=" * 100)
    # If the classifier were a disguised actionability judgement it would agree with a judge
    # about as often as the two judges agree with each other. They agree 0.199 (kappa) on this
    # band, so near-perfect agreement here would be the tell.
    # Raw agreement is useless here, and the first version of this check printed it anyway:
    # with 5% no_proposal and 15% non-actionable, two INDEPENDENT labellers agree ~85% of the
    # time by chance alone. Kappa is the question this check was actually asking.
    a = [1 if r["contribution"] == "no_proposal" else 0 for r in rows]
    b = [1 if r["non_actionable"] else 0 for r in rows]
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    print(f"  raw agreement with GPT non-actionable: {agree}/{len(rows)} = {agree / len(rows):.0%}")
    print(f"    -- uninformative: marginals are {sum(a)}/{len(a)} and {sum(b)}/{len(b)}")
    print(
        f"  Cohen kappa: {cohens_kappa(a, b):.3f}"
        "   (near 0 = NOT re-judging actionability; near 1 = it is)"
    )
    print("\n  ten papers for hand-checking:")
    for r in rows[:10]:
        print(f"    {r['contribution']:12} judge={r['judge_score']}  {r['title'][:56]}")

    verdict = "WIN" if all(v and v["ratio"] >= RATIO_BAR for v in out.values()) else "NULL"
    print(f"\n  VERDICT: {verdict}   (ratio >= {RATIO_BAR:.1f}x under BOTH judges)")
    if verdict == "WIN":
        print("    Licenses a HELD-OUT confirmation only. §28.4: these data generated H-B.")
    (WORK / "proposes_method.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nWrote {WORK / 'proposes_method.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
