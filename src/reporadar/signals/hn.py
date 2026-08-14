"""Hacker News attention for arXiv papers (Feature 9, attention half).

Uses the free, keyless HN Algolia search API. Three things about this signal were
measured rather than assumed, and each one changed the design:

**1. Coverage is near-zero, and not for the reason the roadmap expected.** Over 340
random arXiv papers published in the last week (and a control set 12 weeks old, so
HN had time to react), **zero** had a story — including 0/40 in cs.LG and 0/30 in
cs.CL. The roadmap predicted an ML/systems bias; the real limit is absolute volume,
since HN surfaces a handful of papers a week across all of science. So this ships as
a *badge* first: when it fires, "Discussed on HN (1351 points)" is genuinely useful,
and the ranking weight (``ranking.w_attention``) is opt-in and documented as sparse.

**2. Algolia's typo tolerance silently returns the wrong paper.** Searching
``2501.12948`` also matches a story about ``2501.16948`` — one digit off, and such a
story can carry hundreds of points, which would attribute someone else's front-page
discussion to this paper. Fixed twice over: ``typoTolerance=false`` on the request,
and a client-side check that the id really appears in the story URL.

**3. ``tags=story`` is not optional.** Without it the API returns comment records
whose ``points``, ``title`` and ``url`` are all ``None``.

Requests are kept to a handful per run by first asking one question per year-month
present in the run ("which arXiv ids from 2607 were on HN at all?") and only then
confirming the few candidates exactly. Algolia caps any single query at 1000 hits,
so a very busy month can truncate — a truncated month yields *no* signal for the
missed papers, which the ranker treats as absent rather than as zero attention.
"""

from __future__ import annotations

import json as json_mod
import logging
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from reporadar.paper_id import dedup_id

logger = logging.getLogger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_URL = "https://news.ycombinator.com/item?id={}"

# The API allows ~10k requests/hour and enforces limits by blacklisting rather than
# by returning 429 — there is nothing to back off from once it happens, so stay well
# under the ceiling by construction instead of reacting.
_MIN_REQUEST_INTERVAL_S = 0.5
_last_request_at = 0.0

# Algolia's hard per-query ceiling.
_MAX_HITS_PER_PAGE = 1000

# Cap the month sweep. One request per distinct YYMM is "a handful" for the default
# 14-day window (1-2 months), but `rr update --foundational` reaches back over all of
# arXiv — hundreds of months, i.e. hundreds of requests against an API that enforces
# its limit by blacklisting the IP with no 429 to back off from. Newest months first,
# since that is where a recent digest's papers are.
MAX_MONTHS_PER_RUN = 24

# Points that count as "fully discussed" for normalization. HN points mean the same
# thing across runs and topics (unlike citation counts), so this scale is absolute,
# NOT pool-relative like normalize_citations/normalize_upvotes. Pool-relative would
# break precisely here: a run typically has 0 or 1 discussed papers, so the pool max
# *is* that paper and every hit would normalize to 1.0 regardless of its points.
REFERENCE_POINTS = 500

# Below this, a submission is noise — posted and ignored. Such papers are treated as
# having NO signal rather than a low one: a paper that was posted and got 2 points is
# not worse than a paper nobody posted, so it must not be scored below one that has
# no HN entry at all (an absent component leaves the weighted sum alone; a low score
# drags the total down).
MIN_POINTS = 10

_MODERN_ID_RE = re.compile(r"^\d{4}\.\d{4,5}")


def _throttle() -> None:
    """Block until at least ``_MIN_REQUEST_INTERVAL_S`` since the last request."""
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _base_id(arxiv_id: str) -> str:
    """Strip a version suffix: ``2501.12948v2`` -> ``2501.12948``.

    Delegates to :func:`reporadar.paper_id.dedup_id`. This was a third rule for one
    invariant — anchored at the end, so it survived the ``split("v")[0]`` failure mode, but
    it would still edit a synthetic ``ss:``/``dblp:`` id that merely ended in a
    version-shaped suffix. Those ids are opaque and must pass through untouched.
    """
    return dedup_id(arxiv_id)


def _search(query: str, hits_per_page: int = 20) -> tuple[list[dict[str, Any]], int]:
    """Run one URL-restricted story search. Returns ``(hits, total_available)``.

    ``total_available`` is Algolia's ``nbHits``, which is how a caller detects
    truncation. ``nbPages`` cannot be used for that: a query returning 1000 of 22448
    hits still reports ``nbPages: 1`` with HTTP 200 and no error of any kind.
    """
    params = urllib.parse.urlencode(
        {
            "query": query,
            # Match only the story URL: free-text matching pulls in every "Show HN:
            # an arXiv tool" post (115 false hits for one id in testing).
            "restrictSearchableAttributes": "url",
            "tags": "story",
            "typoTolerance": "false",
            "hitsPerPage": min(hits_per_page, _MAX_HITS_PER_PAGE),
        }
    )
    url = f"{HN_SEARCH_URL}?{params}"
    _throttle()
    try:
        req = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "RepoRadar/1.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json_mod.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.warning("HN search failed for %r: %s", query, exc)
        return [], 0
    except (ValueError, TypeError) as exc:
        logger.warning("HN search returned unparseable JSON for %r: %s", query, exc)
        return [], 0
    hits = payload.get("hits")
    hits = hits if isinstance(hits, list) else []
    try:
        total = int(payload.get("nbHits") or 0)
    except (TypeError, ValueError):
        total = 0
    return hits, total


def _stories_for(hits: list[dict[str, Any]], base_id: str) -> list[dict[str, Any]]:
    """Keep only hits whose URL really contains *base_id* (defeats typo matches)."""
    return [h for h in hits if base_id in str(h.get("url") or "")]


def _months(arxiv_ids: list[str]) -> set[str]:
    """The distinct ``YYMM`` prefixes present, for the pre-filter query."""
    out = set()
    for arxiv_id in arxiv_ids:
        base = _base_id(arxiv_id)
        if _MODERN_ID_RE.match(base):
            out.add(base[:4])
    return out


def discussed_ids(arxiv_ids: list[str]) -> set[str]:
    """Pre-filter: which of *arxiv_ids* appear on HN at all, one request per month.

    Cheap and deliberately approximate — Algolia truncates a query at 1000 hits, so a
    busy month can hide a paper. Missing means "no signal", never "no attention".
    """
    wanted = {_base_id(a) for a in arxiv_ids}
    found: set[str] = set()
    months = sorted(_months(arxiv_ids), reverse=True)[:MAX_MONTHS_PER_RUN]
    if len(_months(arxiv_ids)) > len(months):
        logger.warning(
            "Hacker News sweep limited to the %d most recent months of %d in this run",
            len(months),
            len(_months(arxiv_ids)),
        )
    for month in months:
        hits, total = _search(f"arxiv.org/abs/{month}", hits_per_page=_MAX_HITS_PER_PAGE)
        if total > len(hits):
            # Say so rather than silently under-reporting: a paper hidden by the cap
            # yields no signal, which the ranker reads as absent (correct) but which
            # would otherwise look like "nobody discussed it".
            logger.warning(
                "HN returned %d of %d stories for %s; some discussions may be missed",
                len(hits),
                total,
                month,
            )
        for hit in hits:
            url = str(hit.get("url") or "")
            if "/abs/" not in url:
                continue
            candidate = _base_id(url.split("/abs/")[-1].split("?")[0].strip("/"))
            if candidate in wanted:
                found.add(candidate)
    return found


def fetch_attention(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return ``{arxiv_id: {"points", "comments", "story_url", "title"}}`` for papers on HN.

    Papers with no story are **omitted**, not returned with zeroes: "HN never
    discussed this" is an absent signal, and the vast majority of papers are in that
    bucket. Keys are the ids as passed in, version suffix included.
    """
    if not arxiv_ids:
        return {}

    by_base: dict[str, list[str]] = {}
    for arxiv_id in arxiv_ids:
        base = _base_id(arxiv_id)
        if _MODERN_ID_RE.match(base):
            by_base.setdefault(base, []).append(arxiv_id)

    if not by_base:
        return {}

    candidates = discussed_ids(list(by_base)) & set(by_base)
    results: dict[str, dict[str, Any]] = {}

    for base in sorted(candidates):
        hits, _total = _search(base)
        stories = _stories_for(hits, base)
        if not stories:
            continue
        # A paper can be submitted several times; the best-performing submission is
        # the one that represents "how much attention did this get".
        best = max(stories, key=lambda h: h.get("points") or 0)
        story_id = str(best.get("objectID") or "")
        entry = {
            "points": int(best.get("points") or 0),
            "comments": int(best.get("num_comments") or 0),
            "story_url": HN_ITEM_URL.format(story_id) if story_id else "",
            "title": str(best.get("title") or ""),
            "submissions": len(stories),
        }
        for original in by_base[base]:
            results[original] = entry

    return results


def normalize_points(points_by_id: dict[str, int]) -> dict[str, float]:
    """Log-scale HN points into ``[0, 1]`` against a fixed reference, not the pool max.

    See :data:`REFERENCE_POINTS`: with 0–1 discussed papers per run, a pool-relative
    scale would hand every hit a 1.0 no matter how small its discussion was. Counts
    below :data:`MIN_POINTS` are omitted entirely rather than scored low.
    """
    denom = math.log1p(REFERENCE_POINTS)
    return {
        arxiv_id: min(1.0, math.log1p(points) / denom)
        for arxiv_id, points in points_by_id.items()
        if points and points >= MIN_POINTS
    }
