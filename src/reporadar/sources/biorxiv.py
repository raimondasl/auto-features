"""bioRxiv/medRxiv as a paper source for biology / bioinformatics repos.

bioRxiv's details API is a *date-interval listing*, not a keyword search, so
``collect_papers`` fetches the recent window once (bounded, paginated) and keeps
papers whose title/abstract matches a query term. Coverage is capped by
``max_pages`` — a large lookback window may truncate; that's an acceptable
secondary-source trade-off (the ranker surfaces the best of what's fetched).
Non-arXiv papers get a synthetic ``biorxiv:<doi>`` id.
"""

from __future__ import annotations

import json as json_mod
import logging
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

BIORXIV_API = "https://api.biorxiv.org/details"
# Bound the window fetch: stop after this many pages regardless of the window size.
_MAX_PAGES = 40


def _request_json(url: str, max_retries: int = 3, base_delay: float = 2.0) -> Any | None:
    """GET a JSON endpoint with retry/backoff; None on failure (graceful)."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                time.sleep(base_delay * (2**attempt))
                continue
            logger.warning("bioRxiv API error: %s", exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                logger.warning("bioRxiv request failed after %d attempts: %s", max_retries, exc)
                return None
    return None


def _normalize(item: dict[str, Any], server: str) -> dict[str, Any] | None:
    """Convert a bioRxiv collection item to the internal paper dict, or None."""
    doi = item.get("doi") or ""
    title = item.get("title") or ""
    if not doi or not title:
        return None
    authors = [a.strip() for a in (item.get("authors") or "").split(";") if a.strip()]
    date = item.get("date") or ""
    published = f"{date}T00:00:00+00:00" if date else datetime.now(UTC).isoformat()
    category = item.get("category") or ""
    return {
        "arxiv_id": f"{server}:{doi}",
        "title": title,
        "authors": authors,
        "abstract": item.get("abstract") or "",
        "categories": [category] if category else [],
        "published": published,
        "updated": None,
        "url": f"https://www.{server}.org/content/{doi}",
        "pdf_url": None,
    }


def fetch_window(
    lookback_days: int = 14, server: str = "biorxiv", max_pages: int = _MAX_PAGES
) -> list[dict[str, Any]]:
    """Fetch papers posted in the last *lookback_days* (bounded by *max_pages*)."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=max(1, lookback_days))
    interval = f"{start.isoformat()}/{end.isoformat()}"

    out: list[dict[str, Any]] = []
    cursor = 0
    for _ in range(max_pages):
        data = _request_json(f"{BIORXIV_API}/{server}/{interval}/{cursor}/json")
        if not data:
            break
        collection = data.get("collection") or []
        if not collection:
            break
        for item in collection:
            normalized = _normalize(item, server)
            if normalized:
                out.append(normalized)
        cursor += len(collection)
        try:
            total = int((data.get("messages") or [{}])[0].get("total", 0) or 0)
        except (ValueError, TypeError):
            total = 0
        if total and cursor >= total:
            break
        time.sleep(0.3)  # politeness
    return out


def collect_papers(
    queries: list[str], lookback_days: int = 14, server: str = "biorxiv"
) -> list[dict[str, Any]]:
    """Fetch the bioRxiv window once and keep papers matching any query term.

    Deduplicated by id. Query terms are lowercased words (len > 2); a paper is
    kept if any term appears in its title or abstract.
    """
    terms = {word for query in queries for word in query.lower().split() if len(word) > 2}
    if not terms:
        return []
    seen: dict[str, dict[str, Any]] = {}
    for paper in fetch_window(lookback_days=lookback_days, server=server):
        aid = paper["arxiv_id"]
        if aid in seen:
            continue
        text = f"{paper['title']} {paper['abstract']}".lower()
        if any(term in text for term in terms):
            seen[aid] = paper
    return list(seen.values())
