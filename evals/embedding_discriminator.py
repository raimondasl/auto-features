"""Does the dense embedding discriminate actionable non-arXiv papers? [NR-58]

NR-42 closed the non-arXiv relevance-filter item on **"no instrument discriminates"**, and
named the condition that would reopen it: *"a genuinely better discriminator … the item is
closed on the absence of an instrument, not the absence of value."* The oracle ceiling it
measured is **+1.38** over the control on OpenAlex, against an MRE of 1.04 — so the value is
there if an instrument exists.

**NR-42 tested two instruments and both are LLM stages**: the actionability gate (gate-3 rate
0.588 among actionable, 0.588 among non-actionable) and the fine-scale rescore (0.842 vs
0.850, the wrong way round). It never tested the **dense embedding** — which is the one
scoring component non-arXiv papers do *not* escape. The original argument for the filter was
that they escape the ranker's **category** component; whether the embedding covers for that
was never asked.

## Why this is not just NR-42 again with a different column

**NR-42's labelled set was range-restricted and its own artifact says so.** Its figures are
over papers the pipeline *admitted* — "it does not measure the signal it carries over the
papers it rejected, which this artifact cannot see" — and its OpenAlex panel had **17**
non-actionable papers, every one of which had already survived ranking and the gate.

The judge cache is a wider and differently-selected sample: every experiment ever run,
including papers no digest ever showed. That is the same property NR-43 warned about when it
stratified the cache *by date* and measured the sampling rather than the product — so this
probe does not read a **rate** off it. AUC is a within-case ordering statistic, and the
question "does this signal order these papers" survives a mixed sample in a way "what
fraction are actionable" does not.

**Both panels are reported.** The wide one for power, and NR-42's own 68 shown papers for a
like-for-like comparison against the gate figure — because "the embedding beats the gate"
computed across two different samples would not be a comparison at all.

## Pre-registered, before any AUC was computed

* **Statistic.** Concordance over **within-case** (actionable, non-actionable) pairs, ties
  counted 0.5. Within-case because similarity is repo-relative: a repository whose embedding
  sits closer to everything would otherwise dominate a pooled ranking, and because a filter
  would have to work per repository anyway.
* **Interval.** Cluster bootstrap **over cases**, not over papers (C-7: a single case's level
  is not a property of the method, and 31 cases contribute very unequal counts).
* **Decision rule.** AUC **>= 0.65 with a 95% CI excluding 0.5** on the wide non-arXiv panel
  reopens the filter item. An interval covering 0.5 closes it on a second instrument, better
  powered than the first.
* **Positive control.** The identical computation over **arXiv** papers in the same cases. It
  is what makes a null readable: an embedding that orders neither is a weak instrument
  everywhere, which is a different finding from one that orders arXiv and fails off it.
* **Prediction.** I expect the wide panel to land **near 0.5–0.6 and cover 0.5**, and the
  arXiv control to sit above it. The embedding is a topical-similarity signal and "on topic"
  is most of what the *pool* already selects for; actionability is a property of what a paper
  *proposes*, which is what the gate reads and a bag-of-abstract vector does not.

## The id rule, which nearly broke this before it ran

Verdicts are stored under `judge._cache_path`'s filename stem, so `cs/0412098v3` is written
`cs_0412098v3` — and `is_arxiv_id("cs_0412098v3")` is **False**. Classifying the cache by
filename therefore counts old-style arXiv ids as non-arXiv, which inflated the first count of
this labelled set from 380 to 471. This module never classifies a stem: it matches the pool's
own `arxiv_id` **through `_judge_stem`** and classifies the id, one rule, delegated (C-14,
C-31).

    uv run python evals/embedding_discriminator.py            # $0, no LLM calls
    uv run python evals/embedding_discriminator.py --report   # $0, re-read the artifact
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from diagnose_pool import JUDGE, _judge_stem  # noqa: E402
from harness import WORK_DIR  # noqa: E402

from reporadar.paper_id import is_arxiv_id  # noqa: E402

OUT = EVALS / "embedding_discriminator.json"
ACTIONABLE = 2
SEED = 20260902
N_BOOT = 2000

# Every frozen pool on disk that carries paper TEXT. More pools locate more verdicts, and
# which one a paper came from does not matter — the text is the text. Recorded so a reader
# can tell how the labelled set was assembled rather than inferring it from the count.
POOL_DIRS = (
    "pool-core25-epmc",
    "pool-core25-openalex",
    "pool-core25-arxiv",
    "pool-cohort3",
    "pool-wemb",
    "pool-epmc-treat",
    "pool-oa-treat",
)

# NR-42's own panel: the non-arXiv papers the two source arms SHOWED, which is the set its
# gate figure (0.588 / 0.588) was computed over.
NR42_ARMS = (
    "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json",
    "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260828T052915Z.json",
)


def _verdicts() -> dict[str, dict[str, int]]:
    """{case: {judge filename stem: score}} for every cached verdict."""
    out: dict[str, dict[str, int]] = {}
    for case_dir in sorted(JUDGE.iterdir()):
        if not case_dir.is_dir():
            continue
        scores: dict[str, int] = {}
        for f in case_dir.glob("*.json"):
            try:
                scores[f.stem] = int(json.loads(f.read_text(encoding="utf-8"))["score"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # a malformed verdict is not a verdict; never guessed at
        if scores:
            out[case_dir.name] = scores
    return out


def labelled_set() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """{case: [paper + judge + is_arxiv]} — every cached verdict whose text is on disk.

    A paper is matched by running the POOL's `arxiv_id` through `_judge_stem`, never by
    parsing the stem back into an id. That direction is one-way on purpose: the stem is a
    filename, and reading an id out of it is what makes `cs/0412098v3` look non-arXiv.
    """
    verdicts = _verdicts()
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    located_from: dict[str, int] = {}
    for pool_dir in POOL_DIRS:
        d = WORK_DIR / pool_dir
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            case = f.stem
            if case not in verdicts:
                continue
            try:
                candidates = json.loads(f.read_text(encoding="utf-8"))["candidates"]
            except (json.JSONDecodeError, KeyError):
                continue
            seen = by_case.setdefault(case, {})
            for c in candidates:
                pid = str(c.get("arxiv_id") or "")
                if not pid or pid in seen:
                    continue
                score = verdicts[case].get(_judge_stem(pid))
                if score is None:
                    continue
                seen[pid] = {
                    "arxiv_id": pid,
                    "title": c.get("title", ""),
                    "abstract": c.get("abstract", ""),
                    "judge": score,
                    "actionable": score >= ACTIONABLE,
                    "is_arxiv": is_arxiv_id(pid),
                    "pool": pool_dir,
                }
                located_from[pool_dir] = located_from.get(pool_dir, 0) + 1
    out = {case: list(papers.values()) for case, papers in by_case.items() if papers}
    n_verdicts = sum(len(v) for v in verdicts.values())
    n_located = sum(len(v) for v in out.values())
    return out, {
        "n_verdicts_cached": n_verdicts,
        "n_located_with_text": n_located,
        "n_cases": len(out),
        "located_from": dict(sorted(located_from.items())),
        # Void, not null: verdicts whose paper text is on no frozen pool are ABSENT from the
        # labelled set, not scored. Said out loud because it is most of them.
        "n_unlocatable": n_verdicts - n_located,
    }


def score_papers(cases: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Attach the ranker's own `embedding_score` to every labelled paper.

    The shipped functions, not a local cosine: `compute_repo_embedding`,
    `compute_paper_embedding` and `cosine_similarity`, composed exactly as
    `ranker.rank_papers` composes them, floor at 0 included. A probe that rebuilds the score
    measures the probe (C-3), and this one is testing whether the PRODUCT's signal
    discriminates.
    """
    from reporadar.embeddings import (
        EMBEDDINGS_AVAILABLE,
        compute_paper_embedding,
        compute_repo_embedding,
        cosine_similarity,
    )

    if not EMBEDDINGS_AVAILABLE:
        raise SystemExit(
            "the `embeddings` extra is not installed; refusing to report an instrument's "
            "discrimination from a signal that never ran"
        )
    out: dict[str, list[dict[str, Any]]] = {}
    for case, papers in sorted(cases.items()):
        repo = WORK_DIR / case
        repo_emb = compute_repo_embedding(repo) if repo.is_dir() else None
        if repo_emb is None:
            continue  # no repo text to embed -> the signal is inert here, so it is absent
        scored = []
        for p in papers:
            emb = compute_paper_embedding(p)
            scored.append({**p, "embedding_score": max(0.0, cosine_similarity(repo_emb, emb))})
        out[case] = scored
    return out


def auc_within_case(cases: dict[str, list[dict[str, Any]]], key: str) -> dict[str, Any]:
    """Concordance over within-case (actionable, non-actionable) pairs. Ties count 0.5.

    Pooled across cases by summing pair counts rather than averaging per-case AUCs: a case
    with one comparable pair should not weigh as much as one with four hundred.
    """
    concordant = 0.0
    pairs = 0
    per_case: dict[str, dict[str, Any]] = {}
    for case, papers in sorted(cases.items()):
        pos = [p[key] for p in papers if p["actionable"]]
        neg = [p[key] for p in papers if not p["actionable"]]
        if not pos or not neg:
            continue
        c = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
        concordant += c
        pairs += len(pos) * len(neg)
        per_case[case] = {
            "n_actionable": len(pos),
            "n_non_actionable": len(neg),
            "auc": round(c / (len(pos) * len(neg)), 3),
        }
    return {
        "auc": round(concordant / pairs, 4) if pairs else None,
        "n_pairs": pairs,
        "n_cases_with_both_classes": len(per_case),
        "n_actionable": sum(1 for ps in cases.values() for p in ps if p["actionable"]),
        "n_non_actionable": sum(1 for ps in cases.values() for p in ps if not p["actionable"]),
        "per_case": per_case,
    }


def bootstrap_auc(cases: dict[str, list[dict[str, Any]]], key: str) -> list[float]:
    """95% CI by resampling CASES, not papers.

    Papers within a repository are not independent draws — they share a pool, a profile and a
    repo embedding — and 31 cases contribute counts ranging over two orders of magnitude. A
    paper-level bootstrap would report an interval for a study nobody ran.
    """
    usable = [
        c
        for c, ps in cases.items()
        if any(p["actionable"] for p in ps) and any(not p["actionable"] for p in ps)
    ]
    if len(usable) < 2:
        return []
    rng = random.Random(SEED)
    draws = []
    for _ in range(N_BOOT):
        picked = [usable[rng.randrange(len(usable))] for _ in usable]
        res = auc_within_case({f"{c}#{i}": cases[c] for i, c in enumerate(picked)}, key)
        if res["auc"] is not None:
            draws.append(res["auc"])
    draws.sort()
    if not draws:
        return []
    return [round(draws[int(0.025 * len(draws))], 4), round(draws[int(0.975 * len(draws))], 4)]


def _nr42_panel(scored: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """NR-42's own set: the non-arXiv papers the two source arms SHOWED.

    Restricted to papers this module already located, so the embedding score is the same
    number computed the same way — the point of the panel is a like-for-like comparison
    against a gate figure, and re-deriving either side would defeat it.
    """
    shown: set[tuple[str, str]] = set()
    for name in NR42_ARMS:
        path = EVALS / "results" / name
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            for p in row["returned"]["reporadar_toppicks"]:
                shown.add((row["case"], str(p["arxiv_id"])))
    out: dict[str, list[dict[str, Any]]] = {}
    for case, papers in scored.items():
        keep = [p for p in papers if not p["is_arxiv"] and (case, p["arxiv_id"]) in shown]
        if keep:
            out[case] = keep
    return out


def build() -> dict[str, Any]:
    cases, provenance = labelled_set()
    scored = score_papers(cases)

    def split(pred) -> dict[str, list[dict[str, Any]]]:
        out = {c: [p for p in ps if pred(p)] for c, ps in scored.items()}
        return {c: ps for c, ps in out.items() if ps}

    panels = {
        # The registered primary.
        "non_arxiv_wide": split(lambda p: not p["is_arxiv"]),
        # The positive control. Same signal, same cases, papers the pool was built around.
        "arxiv_control": split(lambda p: p["is_arxiv"]),
        # NR-42's own 68, for the like-for-like read against its gate figure.
        "non_arxiv_nr42_shown": _nr42_panel(scored),
    }
    results = {}
    for name, panel in panels.items():
        res = auc_within_case(panel, "embedding_score")
        res["ci95"] = bootstrap_auc(panel, "embedding_score")
        res["excludes_half"] = bool(res["ci95"] and (res["ci95"][0] > 0.5 or res["ci95"][1] < 0.5))
        sims = [p["embedding_score"] for ps in panel.values() for p in ps]
        act = [p["embedding_score"] for ps in panel.values() for p in ps if p["actionable"]]
        non = [p["embedding_score"] for ps in panel.values() for p in ps if not p["actionable"]]
        res["mean_similarity"] = round(st.mean(sims), 4) if sims else None
        res["mean_actionable"] = round(st.mean(act), 4) if act else None
        res["mean_non_actionable"] = round(st.mean(non), 4) if non else None
        results[name] = res

    primary = results["non_arxiv_wide"]
    reopens = bool(primary["auc"] and primary["auc"] >= 0.65 and primary["excludes_half"])
    return {
        "_comment": (
            "NR-58: does the ranker's dense embedding discriminate actionable non-arXiv "
            "papers, where NR-42's two LLM instruments could not? Derived by "
            "evals/embedding_discriminator.py at $0 from the judge verdict cache and the "
            "frozen pools (both gitignored); pinned by tests/test_embedding_discriminator.py. "
            "No LLM or judge calls. AUC is concordance over WITHIN-CASE pairs; the interval "
            "is a cluster bootstrap over cases."
        ),
        "pre_registered": {
            "written_before_any_auc_was_computed": True,
            "reopens_filter_item_if_auc_at_least": 0.65,
            "and_ci_excludes": 0.5,
            "statistic": "within-case concordance, ties 0.5",
            "interval": "cluster bootstrap over cases",
            "prediction": (
                "wide panel near 0.5-0.6 covering 0.5, arXiv control above it: the embedding "
                "is a topical-similarity signal and actionability is a property of what a "
                "paper PROPOSES, which the gate reads and a bag-of-abstract vector does not"
            ),
            "nr42_gate_on_its_own_panel": {"actionable": 0.588, "non_actionable": 0.588},
            "nr42_oracle_ceiling_openalex": 1.38,
            "benchmark_mre": 1.04,
        },
        "provenance": provenance,
        "panels": results,
        "verdict": {
            "reopens_filter_item": reopens,
            "headline": (
                f"non-arXiv AUC {primary['auc']} over {primary['n_pairs']} within-case pairs "
                f"({primary['n_actionable']} actionable vs {primary['n_non_actionable']} not), "
                f"CI {primary['ci95']}"
            ),
        },
    }


def show(art: dict[str, Any]) -> None:
    p = art["provenance"]
    print(
        f"labelled set: {p['n_located_with_text']} of {p['n_verdicts_cached']} cached "
        f"verdicts located with text, {p['n_cases']} cases"
    )
    print(f"{'panel':<24}{'AUC':>7}{'95% CI':>18}{'pairs':>8}{'act':>6}{'non':>6}")
    for name, r in art["panels"].items():
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]" if r["ci95"] else "n/a"
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "n/a"
        print(
            f"{name:<24}{auc:>7}{ci:>18}{r['n_pairs']:>8}"
            f"{r['n_actionable']:>6}{r['n_non_actionable']:>6}"
        )
    print()
    print("reopens the filter item:", art["verdict"]["reopens_filter_item"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Does the embedding discriminate? $0.")
    ap.add_argument("--report", action="store_true", help="re-read the committed artifact")
    args = ap.parse_args()
    art = json.loads(OUT.read_text(encoding="utf-8")) if args.report else build()
    if not args.report:
        OUT.write_text(json.dumps(art, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT.name}")
    show(art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
