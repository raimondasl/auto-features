"""Audit C-9: what the broken query transform actually did to each non-arXiv source.

`collector.to_plain_keywords` replaced a hand-rolled bridge between arXiv's boolean query
grammar and the plain keyword strings every other source expects. This script measures the
difference the repair makes, per source, **against real `build_queries` output** — the same
discipline the unit tests use, and for the same reason: a hand-written query is exactly what
kept the bug invisible for six months.

It costs nothing. No LLM is called and every API used here is free.

Two properties matter more than the numbers:

**A refused request is not an empty result.** DBLP drops connections under bursts and keyless
Semantic Scholar answers 429 for long stretches. A first pass at this audit reported "0 hits,
old and new alike" for DBLP and nearly published it — 12 of its 18 requests had been refused.
So every request here is spaced, retried, and reported: ``None`` means the source never
answered, and only an integer is a measurement. This project has shipped that mistake before
(seven pools cached empty after an arXiv 429 storm, scored as honest zeros).

**The two failure modes are opposite.** DBLP and IACR return *nothing* for a malformed query.
bioRxiv returns *everything*, because it filters locally on any query word longer than two
characters and the surviving word is the boolean operator ``AND``. A source that goes quiet
announces itself; a source that opens the floodgates looks like it is working.

Usage::

    uv run python evals/audit_query_transform.py                 # dblp + biorxiv
    uv run python evals/audit_query_transform.py --sources s2    # needs an S2 key in practice
    uv run python evals/audit_query_transform.py --cases db,compiler
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from reporadar.collector import build_queries, to_plain_keywords  # noqa: E402
from reporadar.config import ArxivConfig, ProfilerConfig, QueriesConfig  # noqa: E402
from reporadar.profiler import profile_repo  # noqa: E402

WORK = REPO_ROOT / "evals" / ".work"

# DBLP's own adapter uses 1.5 s; that was not enough to keep this audit's zeros honest, and
# an audit that cannot tell refusal from emptiness is worth less than no audit.
DBLP_SPACING_S = 10.0
S2_SPACING_S = 30.0

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def old_transform(query: str) -> str:
    """The bridge every non-arXiv source used before `to_plain_keywords`.

    Kept here rather than imported because it exists nowhere in the codebase any more —
    reproducing the audit requires reproducing the bug.
    """
    return query.replace("all:", "").strip('"')


def real_queries(case: str, limit: int = 3) -> list[str]:
    """Genuine `build_queries` output for a benchmark repo, not a hand-written string."""
    repo = WORK / case
    if not repo.is_dir():
        raise SystemExit(
            f"No clone at {repo}. Run the benchmark once first, or pass --cases with "
            f"names that exist under {WORK}."
        )
    profile = profile_repo(repo, ProfilerConfig())
    return build_queries(profile, QueriesConfig(), ArxivConfig(), max_auto_queries=5)[:limit]


def _get_json(url: str, spacing: float, tries: int) -> Any | None:
    """A parsed response, or None if the host never answered. Never an empty stand-in."""
    for attempt in range(tries):
        time.sleep(spacing)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "RepoRadar-audit (+github.com/raimondasl/auto-features)"},
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"      (attempt {attempt + 1}/{tries} refused: {getattr(exc, 'code', exc)})")
    return None


def dblp_hits(query: str, tries: int = 4) -> int | None:
    from reporadar.sources.dblp import DBLP_SEARCH_URL

    params = urllib.parse.urlencode({"q": query, "format": "json", "h": 30})
    data = _get_json(f"{DBLP_SEARCH_URL}?{params}", DBLP_SPACING_S, tries)
    if data is None:
        return None
    return int(((data.get("result") or {}).get("hits") or {}).get("@total", 0) or 0)


def s2_hits(query: str, tries: int = 5) -> int | None:
    params = urllib.parse.urlencode({"query": query, "limit": 20, "fields": "title"})
    data = _get_json(f"{S2_API}?{params}", S2_SPACING_S, tries)
    return None if data is None else int(data.get("total", 0) or 0)


def audit_keyword_source(name: str, ask: Any, cases: list[str]) -> None:
    """Sources that take a query string: compare what each transform brings back."""
    print(f"\n{'=' * 78}\n{name}: hits reported, old transform vs new")
    print("None = the source never answered; only an integer is a measurement.")
    print("=" * 78)
    for case in cases:
        query = real_queries(case, limit=1)[0]
        old, new = old_transform(query), to_plain_keywords(query)
        print(f"\n--- {case} ---\n  OLD {old!r}")
        old_n = ask(old)
        print(f"    -> {old_n}")
        print(f"  NEW {new!r}")
        new_n = ask(new)
        print(f"    -> {new_n}")


def audit_biorxiv(cases: list[str], lookback_days: int = 7, max_pages: int = 3) -> None:
    """bioRxiv filters locally, so the window is fetched once and re-filtered offline.

    That also means the comparison is exact rather than sampled: both transforms are scored
    against the identical set of papers.
    """
    from reporadar.sources import biorxiv

    print(f"\n{'=' * 78}\nbioRxiv: papers kept from one window, old transform vs new\n{'=' * 78}")
    window = biorxiv.fetch_window(lookback_days=lookback_days, max_pages=max_pages)
    if not window:
        print("bioRxiv returned an empty window — not a result, a failed fetch. Aborting.")
        return
    print(f"window: {len(window)} papers ({lookback_days} days, {max_pages} pages)")
    texts = [f"{p['title']} {p['abstract']}".lower() for p in window]

    def kept(queries: list[str]) -> tuple[int, set[str]]:
        # Mirrors biorxiv.collect_papers exactly; keep the two in step if that changes.
        terms = {w for q in queries for w in q.lower().split() if len(w) > 2}
        return sum(1 for t in texts if any(term in t for term in terms)), terms

    for case in cases:
        queries = real_queries(case)
        old_n, old_terms = kept([old_transform(q) for q in queries])
        new_n, new_terms = kept([to_plain_keywords(q) for q in queries])
        print(f"\n--- {case} ---")
        print(f"  OLD kept {old_n:3d}/{len(window)}   terms: {sorted(old_terms)}")
        print(f"  NEW kept {new_n:3d}/{len(window)}   terms: {sorted(new_terms)}")
        if old_n == len(window):
            print("  ^ the old transform kept the ENTIRE window: the filter was disabled.")
            for term in sorted(old_terms):
                hits = sum(1 for t in texts if term in t)
                print(f"      {term!r:22s} {hits:3d}/{len(window)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="db,columnar,compiler",
        help="Benchmark case names cloned under evals/.work (default: db,columnar,compiler)",
    )
    parser.add_argument(
        "--sources",
        default="dblp,biorxiv",
        help="Comma-separated: dblp, biorxiv, s2 (default: dblp,biorxiv)",
    )
    args = parser.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    unknown = sources - {"dblp", "biorxiv", "s2"}
    if unknown:
        raise SystemExit(f"Unknown source(s): {', '.join(sorted(unknown))}")

    print("Transforms, on real build_queries output:")
    for case in cases:
        for query in real_queries(case, limit=1):
            print(f"  {case}: {query}\n    OLD {old_transform(query)!r}")
            print(f"    NEW {to_plain_keywords(query)!r}")

    if "dblp" in sources:
        audit_keyword_source("DBLP", dblp_hits, cases)
    if "biorxiv" in sources:
        audit_biorxiv(cases)
    if "s2" in sources:
        print("\nNote: keyless Semantic Scholar refused all 20 requests when this was written.")
        audit_keyword_source("Semantic Scholar", s2_hits, cases)


if __name__ == "__main__":
    main()
