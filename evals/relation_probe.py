"""Stage-1 probe, $0: is there a RELATION to classify, and can it be grounded?

Roadmap item 16 promised alerts of the shape *"this paper claims 3x over IVF-PQ, and you
import faiss"*. The 2026-08-09 re-derivation kept its premise -- it seeds from what the
repository HAS, the only register that ever worked here (P2) -- and cut it in half: HyDE
and the citation hop already retrieve these papers, so what is left is the **relation
classification** with a quoted evidence span.

That remainder rests on an assumption nobody has checked: that the relation is *there to
be found and quoted*. This probe checks it before any of it is built, using only cached
artifacts -- no network, no LLM, no judge. It answers four questions, ordered so that a
failure high in the list makes the ones below it moot:

  Q1  Does the judge's own reasoning name a component the repository HAS? If the
      justifications for actionable papers rarely mention anything in the repo's
      vocabulary, there is no X for "supersedes X" to point at.
  Q2  Is there an evidence span to quote? The claim needs a literal span from the
      ABSTRACT. If abstracts never name what the repo imports, the span cannot be found
      by grounding -- it would have to be LLM prose with nothing anchoring it, which is
      the hallucination risk with the mitigation removed.
  Q3  Does the relation vocabulary discriminate? If improves/replaces/extends appears as
      often for non-actionable papers as for actionable ones, the classification carries
      no information the gate does not already have.
  Q4  Coverage. On what fraction of SHOWN papers could a grounded claim be made at all?
      A badge that fires on 5% of entries is not a feature.

**Anchors and keywords are counted separately, and the difference is the point.** An
*anchor* is a package the repository actually depends on, parsed from its manifests -- the
`faiss` in "you import faiss". A *keyword* is a README term, which the retrieval stack
already matches on. A hit on a keyword means "this paper is about your topic", which the
digest implies merely by showing it; only an anchor hit supports the claim item 16 is for.

PRE-REGISTERED, written before running (evals/RESULTS.md keeps the record):
  Q1  HIGH, >60% -- the judge's rubric asks whether a paper improves THIS repo, so its
      justifications should name the repo's components.
  Q2  LOW, 10-20% for anchors -- papers name TECHNIQUES ("IVF-PQ", "late interaction"),
      not the packages that implement them. If this is where it fails, the alias table
      the July entry called a curation burden is not a detail; it is the feature.
  Q3  WEAKLY discriminative -- the judge scores actionability, not relation.
  Q4  20-40%.

    uv run python evals/relation_probe.py        # $0, no network, no LLM, no judge
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import profile_case_repo  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
POOL = EVALS / ".work" / "pool-wemb"
JUDGE = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
WORK = EVALS / ".work"

# The judge's actionable bar, and the metric's own breakeven.
ACTIONABLE = 2

# Relation verbs the feature would classify into. Deliberately the July entry's four, not
# a set tuned after seeing the data.
RELATION_VERBS = {
    "improves": ("improve", "improves", "improving", "outperform", "outperforms", "faster"),
    "replaces": ("replace", "replaces", "instead of", "alternative to", "supersede"),
    "extends": ("extend", "extends", "extending", "builds on", "generalizes", "generalises"),
    "uses": ("uses", "using", "based on", "built on", "leverages"),
}

# Anchors too generic to support a claim: matching them says nothing about a technique.
# Fixed in advance rather than pruned once the counts were visible.
STOPWORD_ANCHORS = frozenset(
    {
        "python",
        "setuptools",
        "wheel",
        "pip",
        "pytest",
        "numpy",
        "scipy",
        "typing",
        "requests",
        "click",
        "pyyaml",
        "six",
        "packaging",
        "attrs",
        "tqdm",
        "rich",
        "black",
        "ruff",
        "mypy",
        "flake8",
        "isort",
        "coverage",
        "sphinx",
        "pandas",
        "matplotlib",
        "setuptools_scm",
        "cmake",
        "ninja",
        "pybind11",
        "cython",
    }
)

MIN_ANCHOR_LEN = 3


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", text.lower()))


def _mentions(text: str, terms: set[str]) -> set[str]:
    """Terms appearing in *text*.

    Single words match as whole tokens, so `ann` does not match `annotation`. The profiler
    also emits multi-word terms ("similarity search"), which have no token to match, so
    those are checked as substrings with word boundaries.
    """
    if not text or not terms:
        return set()
    low = text.lower()
    present = _tokens(low)
    hits = set()
    for t in terms:
        if " " in t:
            if re.search(rf"\b{re.escape(t)}\b", low):
                hits.add(t)
        elif t in present:
            hits.add(t)
    return hits


def _pool_abstracts(case: str) -> dict[str, dict[str, Any]]:
    path = POOL / f"{case}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {dedup_id(c["arxiv_id"]): c for c in data.get("candidates", [])}


def _verdicts(case: str) -> dict[str, dict[str, Any]]:
    d = JUDGE / case
    if not d.is_dir():
        return {}
    out = {}
    for f in d.glob("*.json"):
        out[dedup_id(f.stem)] = json.loads(f.read_text(encoding="utf-8"))
    return out


def _repo_terms(case: str) -> tuple[set[str], set[str]]:
    """(anchors, keywords) for one benchmark repo, from the SHIPPED profiler.

    Not a second corpus implementation: a probe describing a repository nobody profiles is
    the C-9/C-12/C-14 shape this project has corrected three times.
    """
    repo_dir = WORK / case
    if not repo_dir.is_dir():
        return set(), set()
    profile = profile_case_repo(repo_dir)
    anchors = {
        a.lower()
        for a in profile.anchors
        if len(a) >= MIN_ANCHOR_LEN and a.lower() not in STOPWORD_ANCHORS
    }
    # `profile.keywords` is a list of (term, weight) PAIRS, not strings. The first version
    # of this probe treated them as strings, so `len(k) >= 3` measured the tuple's length
    # (2) and discarded every keyword -- and the probe then reported 0.0% keyword hits in
    # all three strata rather than failing. Void read as null, in the probe written to
    # check for exactly that. The `_no_empty_class` guard below is the repair.
    keywords = {
        term.lower()
        for term, _weight in profile.keywords
        if len(term) >= MIN_ANCHOR_LEN and term.lower() not in STOPWORD_ANCHORS
    }
    return anchors, keywords - anchors


def _relations(text: str) -> set[str]:
    low = text.lower()
    return {name for name, verbs in RELATION_VERBS.items() if any(v in low for v in verbs)}


def _no_empty_class(per_case: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Refuse to report a term class that was extracted as empty everywhere.

    A class that produced no terms scores 0.0% in every stratum, which reads as the
    finding "abstracts never mention these" rather than the truth "the extractor returned
    nothing". That is exactly how the tuple-vs-string bug above survived its first run,
    and it is the same shape as the pool scanner that read 1,250 papers as 0.
    """
    for label, key in (("anchors", "anchors"), ("keywords", "keywords")):
        if per_case and sum(c[key] for c in per_case) == 0:
            raise SystemExit(
                f"every case extracted 0 {label}: the extractor is broken, not the corpus. "
                "Reporting 0% hits here would be void read as null."
            )
    if rows and not any(r["relations"] for r in rows):
        raise SystemExit("no relation verb matched any abstract; the matcher is broken")


def main() -> int:
    cases = sorted(p.stem for p in POOL.glob("*.json"))
    if not cases:
        raise SystemExit(f"no frozen pools under {POOL}; this probe reads cached artifacts only")

    rows: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    for case in cases:
        anchors, keywords = _repo_terms(case)
        abstracts = _pool_abstracts(case)
        verdicts = _verdicts(case)
        judged = 0
        for pid, verdict in verdicts.items():
            paper = abstracts.get(pid)
            if paper is None:
                continue  # judged in some other run; no abstract in THIS pool
            judged += 1
            abstract = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            justification = verdict.get("justification") or ""
            rows.append(
                {
                    "case": case,
                    "score": int(verdict.get("score", 0)),
                    "just_anchors": _mentions(justification, anchors),
                    "just_keywords": _mentions(justification, keywords),
                    "abs_anchors": _mentions(abstract, anchors),
                    "abs_keywords": _mentions(abstract, keywords),
                    "relations": _relations(abstract),
                    "has_change": bool((verdict.get("proposed_change") or "").strip()),
                }
            )
        per_case.append(
            {"case": case, "anchors": len(anchors), "keywords": len(keywords), "judged": judged}
        )

    _no_empty_class(per_case, rows)

    good = [r for r in rows if r["score"] >= ACTIONABLE]
    bad = [r for r in rows if r["score"] < ACTIONABLE]
    top = [r for r in rows if r["score"] >= 3]

    def pct(sub: list[dict[str, Any]], key: str) -> float:
        return 100 * sum(1 for r in sub if r[key]) / len(sub) if sub else 0.0

    print("Relation probe -- $0, cached artifacts only\n")
    print(f"  {len(cases)} cases, {len(rows)} judged papers with an abstract in the pool")
    print(f"  {len(good)} actionable (>= {ACTIONABLE}), {len(top)} at 3, {len(bad)} below\n")
    anchors_n = [c["anchors"] for c in per_case]
    print(
        f"  repo anchors per case: median {statistics.median(anchors_n):.0f}, "
        f"min {min(anchors_n)}, max {max(anchors_n)}; "
        f"{sum(1 for n in anchors_n if n == 0)} case(s) with none"
    )
    print()

    print("Q1  Does the judge's reasoning name something the repo HAS?  (predicted >60%)\n")
    print(f"  {'papers':22} {'anchor':>8} {'keyword':>9} {'either':>8}")
    for label, sub in (("score 3", top), ("actionable >=2", good), ("below 2", bad)):
        either = (
            100 * sum(1 for r in sub if r["just_anchors"] or r["just_keywords"]) / len(sub)
            if sub
            else 0.0
        )
        print(
            f"  {label:22} {pct(sub, 'just_anchors'):>7.1f}% "
            f"{pct(sub, 'just_keywords'):>8.1f}% {either:>7.1f}%"
        )
    print()

    print("Q2  Is there a span to QUOTE? Does the ABSTRACT name it?  (predicted 10-20% anchor)\n")
    print(f"  {'papers':22} {'anchor':>8} {'keyword':>9} {'either':>8}")
    for label, sub in (("score 3", top), ("actionable >=2", good), ("below 2", bad)):
        either = (
            100 * sum(1 for r in sub if r["abs_anchors"] or r["abs_keywords"]) / len(sub)
            if sub
            else 0.0
        )
        print(
            f"  {label:22} {pct(sub, 'abs_anchors'):>7.1f}% "
            f"{pct(sub, 'abs_keywords'):>8.1f}% {either:>7.1f}%"
        )
    print()

    print("Q3  Do the relation verbs discriminate?  (predicted: weakly)\n")
    print(f"  {'relation':14} {'actionable':>11} {'below':>9} {'lift':>8}")
    for name in RELATION_VERBS:
        g = 100 * sum(1 for r in good if name in r["relations"]) / len(good) if good else 0.0
        b = 100 * sum(1 for r in bad if name in r["relations"]) / len(bad) if bad else 0.0
        print(f"  {name:14} {g:>10.1f}% {b:>8.1f}% {g - b:>+7.1f}pt")
    any_g = 100 * sum(1 for r in good if r["relations"]) / len(good) if good else 0.0
    any_b = 100 * sum(1 for r in bad if r["relations"]) / len(bad) if bad else 0.0
    print(f"  {'ANY':14} {any_g:>10.1f}% {any_b:>8.1f}% {any_g - any_b:>+7.1f}pt")
    print()

    print("Q4  Coverage: a GROUNDED claim needs an anchor in the abstract AND a relation\n")
    for label, sub in (("score 3", top), ("actionable >=2", good)):
        grounded = [r for r in sub if r["abs_anchors"] and r["relations"]]
        loose = [r for r in sub if (r["abs_anchors"] or r["abs_keywords"]) and r["relations"]]
        print(
            f"  {label:16} anchor+relation {100 * len(grounded) / len(sub) if sub else 0:>5.1f}%"
            f"   |  any-term+relation {100 * len(loose) / len(sub) if sub else 0:>5.1f}%"
        )
    print()

    # Concentration. A mean coverage carried by two repositories is not 12% coverage; it
    # is 100% on two and 0% on the rest, which is a different feature. The thin-docs
    # deficit turned out to rest on one repository, and only the per-case split showed it.
    print("Where the grounded claims actually are (actionable papers per case):\n")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in good:
        by_case.setdefault(r["case"], []).append(r)
    scored = sorted(
        (
            (c, len([r for r in rs if r["abs_anchors"] and r["relations"]]), len(rs))
            for c, rs in by_case.items()
        ),
        key=lambda t: -t[1],
    )
    carried = sum(n for _, n, _ in scored)
    for case, n, total in scored[:6]:
        share = 100 * n / carried if carried else 0.0
        print(
            f"  {case:12} {n:>3}/{total:<3} grounded  ({100 * n / total:>5.1f}% of its own,"
            f" {share:>5.1f}% of all grounded claims)"
        )
    zero = [c for c, n, _ in scored if n == 0]
    print(f"  ...{len(zero)} case(s) with NO grounded claim at all: {zero}")
    print()

    print("What the judge already writes, free:\n")
    print(f"  proposed_change present on {pct(good, 'has_change'):.1f}% of actionable papers")
    top_anchors = Counter(a for r in good for a in r["abs_anchors"]).most_common(8)
    print(f"  most-quoted anchors in abstracts: {top_anchors or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
