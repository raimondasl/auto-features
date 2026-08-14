"""Stage-1, $0: can OpenAlex papers reach a ranked top-10 at all?

OpenAlex is the last of the five non-arXiv channels never measured in any form. DBLP was
found returning nothing, bioRxiv returning *everything*, IACR measured at n = 2, and
Semantic Scholar measured three times — but every published statement about OpenAlex is a
statement about an adapter, not about what it delivers. It also spent six months receiving
the malformed bridge query (C-9), so nothing before 2026-08-12 says anything either.

This is the $0 probe that comes before proposing a judged A/B, following the P4 protocol.
The same check, run earlier, would have caught DBLP before four attempts to benchmark it.

Three numbers per case, and the third decides:

1. **arrived** — papers OpenAlex returned.
2. **new** — those not already in the arXiv pool. OpenAlex keeps a paper's arXiv id when it
   has one, so a paper both sources know collapses on merge and adds nothing. Split into
   ``new-arxiv`` (an arXiv paper our own queries missed) and ``oa:`` (genuinely non-arXiv
   content, which is the entire coverage argument for the source).
3. **in top-10** — how many survive ranking against the arXiv pool.

**Refusal is not emptiness.** ``openalex.search_papers`` returns ``[]`` both when the API
refused and when it honestly found nothing — the defect that made a first DBLP measurement
read "0 vs 0" after 12 of 18 requests were rate-limited, and that made a keyless S2 probe
look like a measurement. The product cannot currently tell those apart either (noted in the
write-up as a real defect, not fixed here). This script counts refusals by wrapping the
adapter's own request function, and refuses to report a case that had any.

**Optimistic by construction**, exactly like `s2_yield.py`: no `--rr-hyde` (~100 further
candidates OpenAlex papers would have to outrank) and no triage rerank, because both cost
money. A number that looks weak here is weaker in production.

Usage::

    uv run python evals/openalex_yield.py                 # all benchmark cases
    uv run python evals/openalex_yield.py --cases rag,db  # a subset
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS_DIR))
sys.path.insert(0, str(EVALS_DIR.parent / "src"))

# Importing harness also enables the shared arXiv response cache (see harness.py), so a
# re-run costs 0 arXiv requests rather than 174 — the volume ceiling, not the rate limit,
# is what throttled this machine on 2026-08-12.
from harness import WORK_DIR, load_benchmark, profile_case_repo  # noqa: E402

from reporadar import arxiv_cache  # noqa: E402
from reporadar.collector import (  # noqa: E402
    CollectionError,
    build_queries,
    collect_papers,
    dedup_id,
    to_plain_keywords,
)
from reporadar.config import ArxivConfig, QueriesConfig, RankingConfig  # noqa: E402
from reporadar.ranker import rank_papers  # noqa: E402
from reporadar.retrieval import hybrid_reorder  # noqa: E402
from reporadar.sources import openalex  # noqa: E402

# OpenAlex documents 10 req/s and 100k/day with a key. One request per second is far inside
# that; this probe is not in a hurry and a throttled answer is worthless to it.
REQUEST_INTERVAL_S = 1.0
ALL_TIME_DAYS = 36500
TOP_N = 10


def load_key() -> str | None:
    env = EVALS_DIR / ".env"
    if env.is_file():
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("OPENALEX_API_KEY="):
                return line.partition("=")[2].strip().strip("\"'") or None
    return os.environ.get("OPENALEX_API_KEY") or None


def _title_key(title: str) -> str:
    """A title reduced to a comparison key: lowercase alphanumerics only.

    Deliberately crude. It is used to ask whether a paper is *already in the pool* under a
    different id, where a false match costs a slightly conservative novelty count and a
    false miss overstates the source's coverage. Erring toward matching is the safe
    direction for a claim about how much genuinely new content a channel supplies.
    """
    return "".join(ch for ch in title.lower() if ch.isalnum())


class RequestWatch:
    """Wrap the adapter's request function to count refusals it cannot report itself.

    ``search_papers`` collapses "the API said no" into the same empty list as "the API
    found nothing", so a caller cannot distinguish them — the failure-is-absence shape this
    project has already published two wrong numbers on. Rather than rebuild OpenAlex's URL
    here (a harness reimplementing the product is its own documented mistake), this wraps
    the private ``_request_json`` and records how many calls came back ``None``.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0
        self._original = openalex._request_json

    def __enter__(self) -> RequestWatch:
        def watched(url: str, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            time.sleep(REQUEST_INTERVAL_S)
            result = self._original(url, *args, **kwargs)
            if result is None:
                self.failures += 1
            return result

        openalex._request_json = watched  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc: object) -> None:
        openalex._request_json = self._original  # type: ignore[assignment]


def measure(case: dict[str, Any], key: str | None) -> dict[str, Any] | None:
    """One case: collect both sources, rank together, report OpenAlex's share of the top-10."""
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

    with RequestWatch() as watch:
        oa_papers = openalex.collect_papers(
            [to_plain_keywords(q) for q in queries[:5]],
            lookback_days=ALL_TIME_DAYS,
            api_key=key,
            rate_limit=0.0,  # RequestWatch does the spacing, so it covers retries too
        )
    if watch.failures:
        print(
            f"  {name:11} UNMEASURED — OpenAlex refused {watch.failures}/{watch.calls} "
            f"requests; a zero from a refusal is not a measurement"
        )
        return None

    known = {dedup_id(p["arxiv_id"]) for p in arxiv_papers}
    new = [p for p in oa_papers if dedup_id(p["arxiv_id"]) not in known]
    new_ids = {p["arxiv_id"] for p in new}
    non_arxiv = [p for p in new if p["arxiv_id"].startswith("oa:")]

    # "New" by id is not new by content. OpenAlex mints an `oa:W...` id for any work whose
    # DOI is not an arXiv DOI, and it indexes the *published* version of a preprint under
    # the publisher's DOI — so a paper already in the arXiv pool can arrive again wearing an
    # id nothing can match it against. Counting that here rather than trusting the id keeps
    # the coverage claim honest: this is the same "did the channel actually deliver
    # anything" question that void-vs-null taught, asked one level down.
    known_titles = {_title_key(p.get("title", "")) for p in arxiv_papers}
    known_titles.discard("")
    shadow = [p for p in non_arxiv if _title_key(p.get("title", "")) in known_titles]

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
        "oa_arrived": len(oa_papers),
        "oa_new": len(new),
        "oa_new_non_arxiv": len(non_arxiv),
        "oa_shadow_dupes": len(shadow),
        "oa_in_top10": len(in_top),
        "requests": watch.calls,
        "top10_ids": top,
    }
    print(
        f"  {name:11} arXiv {len(arxiv_papers):4d} | OA {len(oa_papers):3d} arrived, "
        f"{len(new):3d} new ({len(non_arxiv):3d} non-arXiv, {len(shadow):3d} already in "
        f"the pool under another id) | **{len(in_top)}** in top-{TOP_N}"
    )
    return row


def split_appearances(rows: list[dict[str, Any]], controls: set[str]) -> dict[str, Any]:
    """Where a channel's top-10 slots land, which is the statistic that decides.

    Three buckets, because a raw count hides two ways of winning a slot without earning it:

    * **negative controls** — repositories whose correct output is *nothing*, so a slot
      there is a cost, not a gain. The S2 probe found 30% of its appearances here.
    * **thin-pool cases** — a repository whose arXiv pool came in under half the median has
      half the usual competition, and an ordinary paper places on absence of rivals.
    * **on merit** — everything else, and the only bucket a judged A/B could ever convert.

    Separated from the printing because the first verdict this script emitted was computed
    from `cases_with >= n/4` — it cleared the bar by three quarters of a case and said
    "a judged A/B is justified" while the deciding numbers sat unread in the table above it.
    """
    total = sum(r["oa_in_top10"] for r in rows)
    in_controls = sum(r["oa_in_top10"] for r in rows if r["case"] in controls)
    pools = sorted(r["arxiv_pool"] for r in rows)
    median_pool = pools[len(pools) // 2] if pools else 0
    thin = {
        r["case"] for r in rows if r["arxiv_pool"] < median_pool / 2 and r["case"] not in controls
    }
    in_thin = sum(r["oa_in_top10"] for r in rows if r["case"] in thin)
    return {
        "total": total,
        "cases_with": sum(1 for r in rows if r["oa_in_top10"]),
        "in_controls": in_controls,
        "in_thin": in_thin,
        "on_merit": total - in_controls - in_thin,
        "thin_threshold": median_pool / 2,
    }


def summarise(rows: list[dict[str, Any]], controls: set[str], n_cases: int) -> None:
    """Print the summary and the verdict.

    Split out so ``--from-json`` can re-derive both at $0 from a stored run — a verdict is
    a reading of the numbers, and a reading can be wrong while the numbers are right.
    """
    split = split_appearances(rows, controls)
    total_top, cases_with = split["total"], split["cases_with"]
    in_controls, in_thin = split["in_controls"], split["in_thin"]
    median_pool = int(split["thin_threshold"] * 2)

    print("\n" + "-" * 78)
    print(f"cases measured                : {len(rows)} of {n_cases}")
    print(f"mean OA papers arriving       : {statistics.mean(r['oa_arrived'] for r in rows):.1f}")
    print(f"mean NEW after dedup          : {statistics.mean(r['oa_new'] for r in rows):.1f}")
    mean_non_arxiv = statistics.mean(r["oa_new_non_arxiv"] for r in rows)
    print(f"  of which non-arXiv (oa:)    : {mean_non_arxiv:.1f}")
    mean_shadow = statistics.mean(r["oa_shadow_dupes"] for r in rows)
    total_shadow = sum(r["oa_shadow_dupes"] for r in rows)
    print(f"  ...already in the pool      : {mean_shadow:.1f}  ({total_shadow} total)")
    print(f"OA papers reaching a top-{TOP_N}   : {total_top} across {cases_with}/{len(rows)} cases")
    n_controls = len(controls & {r["case"] for r in rows})
    share = f"{in_controls / total_top:.0%}" if total_top else "n/a"
    print(
        f"  in NEGATIVE CONTROLS        : {in_controls}  ({share} of them, "
        f"from {n_controls}/{len(rows)} cases)"
    )
    print(f"  in THIN-POOL cases          : {in_thin}  (arXiv pool < {median_pool // 2})")
    print(f"  elsewhere, on merit         : {split['on_merit']}")
    cache = arxiv_cache.stats()
    print(
        f"arXiv requests saved by cache : {cache['hits']} hits, {cache['misses']} misses, "
        f"{cache['writes']} written"
    )

    print("\n" + "-" * 78)
    earned = total_top - in_controls - in_thin
    if total_top == 0:
        print("VERDICT: no OpenAlex paper reaches any top-10 even under optimistic settings.")
        print("A judged A/B has nothing to measure. Do not spend on it.")
    elif earned <= len(rows) / 5:
        print(
            f"VERDICT: {total_top} appearances, but only {earned} outside the negative "
            f"controls and the thin-pool cases."
        )
        print("Where the channel places is where placing is worthless: repositories whose")
        print("correct output is nothing, and repositories whose arXiv pool was too small to")
        print("compete. A 25-case judged A/B cannot resolve an effect this concentrated, and")
        print("the comparable source that DID compete broadly (S2, 73 appearances across 16")
        print("cases) was measured end to end and did not help. Do not spend on the A/B.")
    else:
        print(f"VERDICT: OpenAlex competes on merit — {earned} appearances outside the")
        print("controls and thin pools. A judged A/B is justified. This count is an upper")
        print("bound: no HyDE, no rerank.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", help="comma-separated subset (default: all)")
    ap.add_argument("--out", default=str(EVALS_DIR / ".work" / "openalex_yield.json"))
    ap.add_argument(
        "--from-json",
        metavar="PATH",
        help="Re-derive the summary and verdict from a stored run, at $0 and with no "
        "requests. The verdict is a reading of the numbers, and a reading can be wrong "
        "while the numbers are right — this one already was.",
    )
    args = ap.parse_args()

    benchmark = load_benchmark()
    controls = {c["name"] for c in benchmark["cases"] if c.get("negative_control")}
    if args.from_json:
        stored = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        print("=" * 78)
        print(f"OPENALEX STAGE-1 YIELD — re-derived from {args.from_json}")
        print("=" * 78)
        summarise(stored, controls, len(stored))
        return 0

    key = load_key()
    if not key:
        print("No OPENALEX_API_KEY in evals/.env.")
        print("OpenAlex has required a key for its full allowance since 2026-02-13; keyless")
        print("callers get a tiny daily allowance and then throttling, and the zeros that")
        print("produces are not measurements. Aborting.")
        return 1

    cases = load_benchmark()["cases"]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = wanted - {c["name"] for c in cases}
        if unknown:
            raise SystemExit(f"Unknown case(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["name"] in wanted]

    print("=" * 78)
    print(f"OPENALEX STAGE-1 YIELD — {len(cases)} cases, $0, no LLM")
    print("Optimistic by construction: no HyDE (~100 more competing candidates) and no")
    print("triage rerank, because both cost money. Treat top-10 counts as an UPPER BOUND.")
    print("A case with any refused request is reported UNMEASURED, never as a zero.")
    print("=" * 78)

    rows = [r for c in cases if (r := measure(c, key)) is not None]
    if not rows:
        print("\nNothing measured.")
        return 1

    summarise(rows, controls, len(cases))
    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
