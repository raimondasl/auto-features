"""Stage-1, $0: can Semantic Scholar papers reach a ranked top-10 at all?

`evals/RESULTS.md` finding 3 ("adding Semantic Scholar did not help") turned out to be
**void** — the arm sent malformed queries and S2 answered with nothing (see *S2 resolved*).
With a key and repaired queries S2 now returns 20 papers per query, so the obvious next step
is the judged A/B that finding 3 only appeared to run. That costs ~$26 and ~4 hours.

This is the $0 check that comes first, following the P4 protocol — verify every dependency
before building. **If S2's papers cannot reach a top-10, the judged experiment has nothing
to measure** and is dead before it spends anything. The same check, run earlier, would have
caught that DBLP returns nothing long before four separate attempts to benchmark it.

Three numbers per case, and the third is the one that decides:

1. **arrived** — papers S2 returned.
2. **new** — those not already in the arXiv pool. S2 keeps a paper's arXiv id when it has
   one, so a paper both sources know collapses on merge and adds nothing. Split into
   ``new-arxiv`` (an arXiv paper our own queries missed) and ``ss:`` (genuinely non-arXiv
   content, which is the coverage argument for the source in the first place).
3. **in top-10** — how many survive ranking against the arXiv pool.

**This is optimistic on purpose.** It omits `--rr-hyde`, which adds ~100 further candidates
that S2 papers would have to outrank, and `--rr-rerank`, because both cost money. So the
top-10 share here is an *upper bound* on what the shipped configuration would show. A number
that looks weak here is weaker in production; a number that looks strong still has to survive
HyDE.

Usage::

    uv run python evals/s2_yield.py                    # all benchmark cases
    uv run python evals/s2_yield.py --cases rag,db     # a subset
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(EVALS_DIR.parent / "src"))

# Importing harness also enables the shared arXiv response cache (see harness.py) — which
# is why this script no longer costs 174 arXiv requests on a re-run.
from harness import WORK_DIR, load_benchmark, profile_case_repo  # noqa: E402

from reporadar import arxiv_cache, s2_rate  # noqa: E402
from reporadar.collector import (  # noqa: E402
    CollectionError,
    build_queries,
    collect_papers,
    to_plain_keywords,
)
from reporadar.config import ArxivConfig, QueriesConfig, RankingConfig  # noqa: E402
from reporadar.paper_id import dedup_id, is_arxiv_id  # noqa: E402
from reporadar.pipeline import KEYWORD_SOURCE_QUERIES  # noqa: E402
from reporadar.ranker import rank_papers  # noqa: E402
from reporadar.retrieval import hybrid_reorder  # noqa: E402
from reporadar.sources.semantic_scholar import collect_papers as s2_collect  # noqa: E402

# Slower than the 1 RPS floor. S2 throttles beyond its documented limit under load, and a
# refusal miscounted as "S2 returned nothing" is exactly the error this script exists to
# avoid making about someone else's source.
S2_INTERVAL_S = 2.0
ALL_TIME_DAYS = 36500
TOP_N = 10


def load_key() -> str | None:
    env = EVALS_DIR / ".env"
    if env.is_file():
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("SEMANTIC_SCHOLAR_API_KEY="):
                return line.partition("=")[2].strip().strip("\"'") or None
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None


def measure(case: dict[str, Any], key: str | None) -> dict[str, Any] | None:
    """One case: collect both sources, rank together, report S2's share of the top-10."""
    name = case["name"]
    repo = WORK_DIR / name
    if not repo.is_dir():
        print(f"  {name:11} no clone at {repo} — skipped")
        return None

    profile = profile_case_repo(repo)
    categories = case.get("expected_categories") or ["cs.LG", "cs.CL", "cs.CV", "cs.SE"]
    arxiv_cfg = ArxivConfig(
        categories=categories,
        max_results_per_query=50,
        lookback_days=ALL_TIME_DAYS,
        sort_by="relevance",
    )
    queries = build_queries(profile, QueriesConfig(), arxiv_cfg)

    try:
        arxiv_papers = collect_papers(queries, arxiv_cfg)
    except CollectionError as exc:
        # Never degrade a failed fetch into "the source found nothing" — that mistake cost
        # this project two benchmark cases scored as honest zeros after an arXiv 429 storm.
        print(f"  {name:11} arXiv collection FAILED (not an empty result): {exc}")
        return None

    s2_papers = s2_collect(
        [to_plain_keywords(q) for q in queries[:KEYWORD_SOURCE_QUERIES]],
        api_key=key,
        lookback_days=ALL_TIME_DAYS,
    )

    known = {dedup_id(p["arxiv_id"]) for p in arxiv_papers}
    new = [p for p in s2_papers if dedup_id(p["arxiv_id"]) not in known]
    new_ids = {p["arxiv_id"] for p in new}
    # Asked positively — see the same line in `openalex_yield.py`. `ss:` stopped being the
    # only non-arXiv id this adapter can mint when F15 made a known DOI the id.
    non_arxiv = [p for p in new if not is_arxiv_id(p["arxiv_id"])]

    merged = arxiv_papers + new
    ranked = rank_papers(
        merged,
        profile,
        RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.0),
        QueriesConfig(),
        categories,
        lookback_days=ALL_TIME_DAYS,
    )
    ranked = hybrid_reorder(ranked, merged, profile)
    top = [r["arxiv_id"] for r in ranked[:TOP_N]]
    in_top = [i for i in top if i in new_ids]

    row = {
        "case": name,
        "arxiv_pool": len(arxiv_papers),
        "s2_arrived": len(s2_papers),
        "s2_new": len(new),
        "s2_new_non_arxiv": len(non_arxiv),
        "s2_in_top10": len(in_top),
        "top10_ids": top,
    }
    print(
        f"  {name:11} arXiv {len(arxiv_papers):4d} | S2 {len(s2_papers):3d} arrived, "
        f"{len(new):3d} new ({len(non_arxiv):3d} non-arXiv) | "
        f"**{len(in_top)}** in top-{TOP_N}"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", help="comma-separated subset (default: all)")
    ap.add_argument("--out", default=str(EVALS_DIR / ".work" / "s2_yield.json"))
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("No SEMANTIC_SCHOLAR_API_KEY in evals/.env.")
        print("Keyless S2 shares one pool with every unauthenticated user and was refused")
        print("20/20 twice — the zeros that produces are not measurements. Aborting.")
        return 1
    s2_rate.set_min_interval(S2_INTERVAL_S)

    cases = load_benchmark()["cases"]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = wanted - {c["name"] for c in cases}
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["name"] in wanted]

    print("=" * 78)
    print(f"S2 STAGE-1 YIELD — {len(cases)} cases, $0, no LLM")
    print("Optimistic by construction: no HyDE (~100 more competing candidates) and no")
    print("triage rerank, because both cost money. Treat top-10 counts as an UPPER BOUND.")
    print("=" * 78)

    rows = [r for c in cases if (r := measure(c, key)) is not None]
    if not rows:
        print("\nNothing measured.")
        return 1

    total_top = sum(r["s2_in_top10"] for r in rows)
    cases_with = sum(1 for r in rows if r["s2_in_top10"])
    print("\n" + "-" * 78)
    print(f"cases measured                : {len(rows)}")
    print(f"mean S2 papers arriving       : {statistics.mean(r['s2_arrived'] for r in rows):.1f}")
    mean_new = statistics.mean(r["s2_new"] for r in rows)
    mean_non_arxiv = statistics.mean(r["s2_new_non_arxiv"] for r in rows)
    print(f"mean NEW after dedup          : {mean_new:.1f}")
    print(f"  of which non-arXiv (ss:)    : {mean_non_arxiv:.1f}")
    print(f"S2 papers reaching a top-{TOP_N}   : {total_top} across {cases_with}/{len(rows)} cases")
    cache = arxiv_cache.stats()
    print(
        f"arXiv requests saved by cache : {cache['hits']} hits, {cache['misses']} misses, "
        f"{cache['writes']} written"
    )

    print("\n" + "-" * 78)
    if total_top == 0:
        print("VERDICT: no S2 paper reaches any top-10 even under optimistic settings.")
        print("The judged A/B has nothing to measure. Do not spend $26 on it.")
    elif cases_with < len(rows) / 4:
        print(f"VERDICT: S2 reaches a top-10 in only {cases_with}/{len(rows)} cases.")
        print("A 25-case mean cannot resolve a channel that touches so few — size any")
        print("experiment on the subset it actually serves, against a PLAUSIBLE effect.")
    else:
        print(f"VERDICT: S2 competes — {total_top} papers across {cases_with} cases.")
        print("The judged A/B is justified. Remember this count is an upper bound.")

    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
