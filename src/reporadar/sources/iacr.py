"""IACR Cryptology ePrint Archive as a paper source for cryptography repositories.

Cryptography's literature is largely **not on arXiv**. It lives on ePrint, which is why
`crypto` and `encryption` have been the benchmark's most persistent under-performers: the
papers that would improve them were never in the candidate pool, and no reranker recovers a
paper that was never fetched.

**Why this is measured on two cases and not on the benchmark mean.** The benchmark's
minimum resolvable effect is 1.04 net@2 per case (`evals/noise_floor.py`). ePrint serves
exactly two of the 25 cases, so even a *perfect* adapter moves the 25-case mean by at most
+0.68 — below the floor, undetectable by construction. On the two-case subset the effect is
measurable, and that subset was pre-registered in `evals/verify_iacr_deps.py` before this
module existed.

**Route, and why not the other one.** ePrint exposes OAI-PMH (verified: Dublin Core with
1,346-character abstracts and stable `2026/1373` identifiers) — but OAI-PMH has no keyword
search, only date-range harvesting, which would mean pulling the whole archive to answer one
query. The HTML search endpoint *is* keyword-addressable, returns 100 results with abstracts
inline in a single ~0.3 s request, and is parsed here off semantic class names
(``paperlink``, ``search-abstract``) rather than document position.

That parse is the fragile part, so it is guarded rather than trusted: a response that yields
zero papers **and** does not carry ePrint's own ``No results`` marker raises
:class:`CollectionError`. "The structure changed" and "there is nothing here" must not
produce the same empty list — this project has already shipped one silent-empty failure
(seven pools cached empty after an arXiv 429 storm, scored as honest zeros).

ePrint is run by a non-profit on donated infrastructure. Requests go through a process-wide
minimum interval, as DBLP's adapter does, and the archive is never paginated beyond the
first page of results.
"""

from __future__ import annotations

import html
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

from reporadar import __version__

logger = logging.getLogger(__name__)

SEARCH_URL = "https://eprint.iacr.org/search"
PAPER_URL = "https://eprint.iacr.org/{paper_id}"
USER_AGENT = f"RepoRadar/{__version__} (+https://github.com/raimondasl/auto-features)"

# A volunteer-run archive. One request per second, process-wide — the eval sweeps many
# repos in one process, so throttling inside a single call would not be enough.
_MIN_REQUEST_INTERVAL_S = 1.0
_last_request_at = 0.0

# ePrint's own wording when a query matches nothing. Its presence is what separates
# "no results" from "the page structure changed underneath us".
NO_RESULTS_MARKER = "No results"

_RESULT_RE = re.compile(
    r'<a\s+title="(?P<pid>\d{4}/\d+)"\s+class="paperlink".*?'
    r"<strong>(?P<title>.*?)</strong>.*?"
    r'(?:<span class="fst-italic">(?P<authors>.*?)</span>)?.*?'
    r'<p class="[^"]*search-abstract"[^>]*>(?P<abstract>.*?)</p>',
    re.S,
)
_UPDATED_RE = re.compile(r"Last updated:\s*(\d{4}-\d{2}-\d{2})")
_TAG_RE = re.compile(r"<[^>]+>")


class CollectionError(Exception):
    """Raised when ePrint could not be read — never to mean 'found nothing'."""


def _throttle() -> None:
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _clean(fragment: str) -> str:
    """Strip tags and unescape entities, collapsing whitespace."""
    return " ".join(html.unescape(_TAG_RE.sub(" ", fragment)).split())


def _fetch(query: str, timeout: int = 30) -> str:
    _throttle()
    url = f"{SEARCH_URL}?{urllib.parse.urlencode({'q': query})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body: str = resp.read().decode("utf-8", errors="replace")
            return body
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CollectionError(f"IACR ePrint search failed for {query!r}: {exc}") from exc


def parse_results(body: str, query: str) -> list[dict[str, Any]]:
    """Papers from one search response.

    Raises when the response carries neither results nor ePrint's no-results marker,
    because that combination means the markup changed and every caller downstream would
    otherwise read a silent zero as a measured one.
    """
    out: list[dict[str, Any]] = []
    # Split on the anchor that opens each result so one malformed block cannot swallow
    # the rest of the page.
    for match in _RESULT_RE.finditer(body):
        paper_id = match.group("pid")
        title = _clean(match.group("title") or "")
        abstract = _clean(match.group("abstract") or "")
        if not title:
            continue
        tail = body[match.end() : match.end() + 400]
        head = body[max(0, match.start() - 400) : match.start() + 400]
        updated = _UPDATED_RE.search(head) or _UPDATED_RE.search(tail)
        out.append(
            {
                # Synthetic id, matching the `dblp:`/`ss:`/`oa:` scheme the pipeline
                # already uses for non-arXiv papers. Nothing downstream may assume an
                # ePrint id resolves at arXiv.
                "arxiv_id": f"iacr:{paper_id}",
                "title": title,
                "abstract": abstract,
                "authors": _clean(match.group("authors") or ""),
                "url": PAPER_URL.format(paper_id=paper_id),
                "published": f"{updated.group(1)}T00:00:00Z" if updated else "",
                "categories": ["cs.CR"],
                "source": "iacr",
            }
        )
    if not out and NO_RESULTS_MARKER not in body:
        raise CollectionError(
            f"IACR ePrint returned {len(body)} bytes for {query!r} with neither results nor "
            f"its {NO_RESULTS_MARKER!r} marker — the page structure has probably changed. "
            "Refusing to report this as 'no papers found'."
        )
    return out


_OG_DESC_RE = re.compile(r'<meta\s+property="og:description"\s+content="(.*?)"\s*/?>', re.S)


def fetch_full_abstract(paper_id: str, timeout: int = 30) -> str | None:
    """The complete abstract from a paper's own page, or None if it cannot be read.

    **The search page truncates.** Measured across six results it caps at 488-499
    characters and stops mid-sentence, while the paper page carries 1,420 — comparable to
    a typical arXiv abstract. That gap is not cosmetic: the 0-3 gate reads
    ``abstract[:1500]``, so ePrint papers judged on search snippets would arrive
    systematically less specific than arXiv ones, and a null result for this source could
    not be told apart from an artefact of its own adapter.

    The full text lives in the ``og:description`` meta tag; the on-page abstract is
    rendered by script, so scraping the body would return nothing.
    """
    _throttle()
    url = PAPER_URL.format(paper_id=paper_id)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("IACR ePrint: could not fetch %s for its full abstract: %s", paper_id, exc)
        return None
    match = _OG_DESC_RE.search(body)
    return _clean(match.group(1)) if match else None


def enrich_abstracts(papers: list[dict[str, Any]], *, limit: int = 150) -> int:
    """Replace truncated search abstracts with full ones, in place. Returns the count.

    Capped because each paper costs one throttled request against a volunteer-run
    archive. Papers beyond *limit* keep their truncated abstract rather than being
    dropped — a shorter abstract is a handicap, a missing paper is a hole.
    """
    enriched = 0
    for paper in papers[:limit]:
        paper_id = paper["arxiv_id"].removeprefix("iacr:")
        full = fetch_full_abstract(paper_id)
        if full and len(full) > len(paper.get("abstract", "")):
            paper["abstract"] = full
            enriched += 1
    if len(papers) > limit:
        logger.info(
            "IACR ePrint: %d of %d papers kept their truncated search abstract (limit %d)",
            len(papers) - limit,
            len(papers),
            limit,
        )
    return enriched


def collect_papers(
    queries: list[str],
    *,
    lookback_days: int = 90,
    max_results_per_query: int = 50,
    full_abstracts: bool = True,
) -> list[dict[str, Any]]:
    """Search ePrint for each query and return de-duplicated papers, newest first.

    *lookback_days* filters on ePrint's "Last updated" date. A paper with no parseable
    date is **kept**: ePrint occasionally omits it, and dropping those would silently
    prefer recently-touched papers in a source whose value is its older, seminal work.
    """
    cutoff = datetime.now(UTC) - timedelta(days=max(1, lookback_days))
    seen: set[str] = set()
    papers: list[dict[str, Any]] = []
    for query in queries:
        try:
            body = _fetch(query)
        except CollectionError as exc:
            logger.warning("%s", exc)
            continue
        for paper in parse_results(body, query)[:max_results_per_query]:
            if paper["arxiv_id"] in seen:
                continue
            if paper["published"]:
                try:
                    when = datetime.fromisoformat(paper["published"].replace("Z", "+00:00"))
                except ValueError:
                    when = None
                if when is not None and when < cutoff:
                    continue
            seen.add(paper["arxiv_id"])
            papers.append(paper)
    papers.sort(key=lambda p: p.get("published") or "", reverse=True)
    enriched = enrich_abstracts(papers) if full_abstracts else 0
    logger.info(
        "IACR ePrint: %d papers from %d queries (%d with full abstracts)",
        len(papers),
        len(queries),
        enriched,
    )
    return papers
