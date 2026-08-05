"""P2: do "what this repo lacks" phrases retrieve the targets out of the hop pool?

§3.2 measured the sharpest fact in this project. Asked what a repo LACKS, the model aims at
exactly the right research — quantization, distillation, ViT backbones, cross-encoder
reranking — and then phrases it as compounds no paper title contains: **45 of 54 phrases
matched zero papers**. Its own example:

    "experience replay prioritization methods"   -> 0 hits   (the "lacks" prompt)
    "prioritized experience replay"              -> FOUND    (the "uses" prompt)

Same concept, same target paper. One is the literature's term of art, the other a description
of it. That is a *string-matching* failure, not a reasoning failure, and §3.2 said so: the
follow-up it named was matching in a space where those two are close.

This tests the cheapest such space first. Stemming collapses the morphology
(`prioritization`/`prioritized` -> `priorit`) and BM25 scores bag-of-terms overlap rather
than phrase containment, so word order and compounding stop mattering. The candidates are
the persisted citation-hop pool, which is where the targets actually are.

    uv run python evals/gap_match.py                # all cases with pool text
    uv run python evals/gap_match.py --top-k 500    # deeper cut

Arms, all ranking the same pool with the same scorer, differing only in the QUERY:

  lacks       the "lacks" phrases  — what the repo should adopt
  uses        the "uses" phrases   — what the repo already implements (cached from P3)
  gaps        the summariser's `improvement_areas` (§5.3), never tested on retrieval
  profile     the repo's keyword profile — the CONTROL. Design 3 reports that
              "similar to what this repo does" does not rank the targets highly, and that
              claim has never been MEASURED. If this arm wins, similarity is not the wrong
              relation and the simpler thing should ship.

Pre-registered (ROADMAP P2, restated for the 21-target pool): the best phrase arm puts
**>=44% of reachable targets** in the top 200/repo, and beats `profile` by >=2x. KILL if no
arm clears the `profile` control at top-500, in which case gap phrases add nothing over
describing the repo and the escalation to dense vectors is the only remaining move.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import diagnose_query_generation as qg  # noqa: E402
import snowballstemmer  # noqa: E402

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402
from reporadar.profiler import _collect_text_corpus, profile_repo  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
POOL_DIR = WORK / "hop_pool"
LACKS_CACHE = WORK / "lacks_phrases.json"
OUT = WORK / "gap_match.json"

N_PHRASES = 10
_STEM = snowballstemmer.stemmer("english")
# Terms that carry no discriminative signal in a pool that is already all CS papers.
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "and",
    "or",
    "to",
    "in",
    "on",
    "with",
    "by",
    "via",
    "using",
    "based",
    "method",
    "methods",
    "approach",
    "approaches",
    "technique",
    "techniques",
    "model",
    "models",
    "system",
    "systems",
    "learning",
    "neural",
    "network",
    "networks",
    "deep",
    "new",
    "novel",
    "improved",
    "efficient",
}


def stem_tokens(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, stem.

    Stemming is the whole point: `prioritization` and `prioritized` are different strings
    and the same concept, and §3.2's zero-hit phrases differ from the papers that answer
    them mostly by inflection and word order.
    """
    raw = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return [s for w in raw if len(w) > 2 and w not in _STOP and (s := _STEM.stemWord(w))]


def bm25(
    corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Okapi BM25 of *query* against each pre-tokenised document."""
    n = len(corpus)
    avgdl = sum(len(d) for d in corpus) / max(n, 1)
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(doc))
    tfs = [Counter(d) for d in corpus]
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(query) if df[t]}
    out = []
    for tf, doc in zip(tfs, corpus, strict=True):
        dl = len(doc)
        s = 0.0
        for t, w in idf.items():
            f = tf.get(t, 0)
            if f:
                s += w * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
        out.append(s)
    return out


def lacks_phrases(case: str, cfg: SuggestionsConfig, cache: dict) -> list[str]:
    if case in cache:
        return cache[case]
    repo = WORK / case
    profile = profile_repo(repo)
    docs = _collect_text_corpus(repo)
    prompt = qg.PROMPTS["lacks"].format(
        n=N_PHRASES,
        anchors=", ".join(profile.anchors[:12]) or "none",
        domains=", ".join(profile.domains[:5]) or "general",
        keywords=", ".join(t for t, _ in profile.keywords[:12]) or "n/a",
        docs=(docs[0] if docs else "")[:1500],
    )
    raw = complete(prompt, cfg, max_tokens=400)
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b < 0:
        raise ValueError(f"no JSON array: {raw[:120]}")
    cache[case] = [str(p).strip() for p in json.loads(raw[a : b + 1]) if str(p).strip()][:N_PHRASES]
    LACKS_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache[case]


def queries_for(case: str, cfg: SuggestionsConfig, lacks_cache: dict) -> dict[str, str]:
    """One query string per arm. Empty string means the arm is unavailable for this case."""
    out: dict[str, str] = {}
    try:
        out["lacks"] = " ".join(lacks_phrases(case, cfg, lacks_cache))
    except Exception as exc:  # noqa: BLE001
        print(f"    ! lacks phrases failed: {exc}")
        out["lacks"] = ""

    uses_file = WORK / "synth_phrases.json"
    uses = json.loads(uses_file.read_text(encoding="utf-8")) if uses_file.is_file() else {}
    out["uses"] = " ".join(uses.get(case, []))

    summ_file = WORK / "repo_summaries.json"
    summ = json.loads(summ_file.read_text(encoding="utf-8")) if summ_file.is_file() else {}
    out["gaps"] = " ".join((summ.get(case) or {}).get("improvement_areas") or [])

    profile = profile_repo(WORK / case)
    out["profile"] = " ".join(
        [t for t, _ in profile.keywords[:12]] + profile.anchors[:12] + profile.domains[:5]
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="skip a case whose pool has text for less than this fraction",
    )
    args = ap.parse_args()
    qg._load_env()

    cache = json.loads((POOL_DIR / "_meta_cache.json").read_text(encoding="utf-8"))
    lacks_cache = (
        json.loads(LACKS_CACHE.read_text(encoding="utf-8")) if LACKS_CACHE.is_file() else {}
    )
    cfg = SuggestionsConfig(provider="claude", claude_model=args.model, timeout=90)

    arms = ("lacks", "uses", "gaps", "profile")
    totals = {a: 0 for a in arms}
    reachable = 0
    rows = []
    skipped = []

    for path in sorted(POOL_DIR.glob("*.jsonl")):
        case = path.stem
        pool = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
        withtext = [r for r in pool if (cache.get(r["id"]) or {}).get("abstract")]
        cov = len(withtext) / max(len(pool), 1)
        tgt = [r["id"] for r in withtext if r["is_target"]]
        if cov < args.min_coverage or not tgt:
            skipped.append((case, cov, len(tgt)))
            continue
        reachable += len(tgt)

        corpus = [
            stem_tokens(
                f"{(cache[r['id']]).get('title', '')} {(cache[r['id']]).get('abstract', '')}"
            )
            for r in withtext
        ]
        qs = queries_for(case, cfg, lacks_cache)
        row = {"case": case, "pool": len(withtext), "coverage": round(cov, 3), "targets": len(tgt)}
        for arm in arms:
            q = stem_tokens(qs.get(arm, ""))
            if not q:
                row[arm] = None
                continue
            scores = bm25(corpus, q)
            top = {
                withtext[i]["id"]
                for i in sorted(range(len(scores)), key=lambda j: -scores[j])[: args.top_k]
            }
            hit = len(set(tgt) & top)
            totals[arm] += hit
            row[arm] = hit
        rows.append(row)
        print(
            f"[{case:10}] pool={len(withtext):6,} ({cov:.0%} text) targets={len(tgt)}  "
            + "  ".join(f"{a}={row[a]}" for a in arms),
            flush=True,
        )

    for case, cov, nt in skipped:
        print(f"[{case:10}] SKIPPED — {cov:.0%} text coverage, {nt} targets with text")

    OUT.write_text(json.dumps({"top_k": args.top_k, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n=== P2 RESULT (top-{args.top_k}/repo, {len(rows)} cases) ===")
    print(f"reachable targets: {reachable}")
    for arm in arms:
        pct = totals[arm] / max(reachable, 1)
        print(f"  {arm:8} {totals[arm]:3}/{reachable}  {pct:5.0%}")
    best = max((a for a in arms if a != "profile"), key=lambda a: totals[a])
    ctrl = totals["profile"]
    print(f"\nbest phrase arm: {best} at {totals[best]}/{reachable}; control (profile) {ctrl}")
    print("PRE-REGISTERED: best >=44% of reachable AND >=2x control; kill if none beats control")
    ok = totals[best] / max(reachable, 1) >= 0.44 and totals[best] >= 2 * max(ctrl, 1)
    if ok:
        verdict = "MET"
    elif totals[best] <= ctrl:
        verdict = "KILL — no phrase arm beats the repo-description control"
    else:
        verdict = "BELOW PREDICTION"
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
