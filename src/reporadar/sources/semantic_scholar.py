"""Semantic Scholar as a secondary paper search source."""

from __future__ import annotations

import json as json_mod
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
SS_SEARCH_FIELDS = "externalIds,title,authors,abstract,publicationTypes,year,url,citationCount"


def _request_json(
    url: str,
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Any | None:
    """GET a JSON endpoint with retry and backoff."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or exc.code >= 500:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Semantic Scholar API error %d (attempt %d/%d). Retrying in %.1fs...",
                    exc.code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("Semantic Scholar API error: %s", exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Semantic Scholar request failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "Semantic Scholar API failed after %d attempts: %s", max_retries, exc
                )
                return None

    logger.warning("Semantic Scholar API failed after %d attempts: %s", max_retries, last_exc)
    return None


def _normalize_paper(paper: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Semantic Scholar paper to internal format."""
    title = paper.get("title", "")
    if not title:
        return None

    # Extract arXiv ID from externalIds
    external_ids = paper.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv", "")
    if not arxiv_id:
        # Use synthetic ID for non-arXiv papers
        paper_id = paper.get("paperId", "")
        if not paper_id:
            return None
        arxiv_id = f"ss:{paper_id}"

    # Authors
    authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]

    # Abstract
    abstract = paper.get("abstract", "") or ""

    # Year → approximate date
    year = paper.get("year")
    published = f"{year}-01-01T00:00:00+00:00" if year else datetime.now(UTC).isoformat()

    # URL
    url = paper.get("url", "")
    if not url and arxiv_id and not arxiv_id.startswith("ss:"):
        url = f"http://arxiv.org/abs/{arxiv_id}"

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "categories": [],  # SS doesn't provide arXiv categories
        "published": published,
        "updated": None,
        "url": url,
        "pdf_url": None,
    }


def search_papers(
    query: str,
    limit: int = 50,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for papers matching a query.

    Returns a list of papers in internal dict format.
    """
    encoded_query = urllib.parse.quote(query)
    url = (
        f"{SS_API_BASE}/paper/search?query={encoded_query}&fields={SS_SEARCH_FIELDS}&limit={limit}"
    )

    data = _request_json(url, api_key=api_key)
    if data is None:
        return []

    results: list[dict[str, Any]] = []
    for paper in data.get("data", []):
        normalized = _normalize_paper(paper)
        if normalized:
            results.append(normalized)

    return results


def collect_papers(
    queries: list[str],
    api_key: str | None = None,
    lookback_days: int = 14,
    rate_limit: float = 1.0,
) -> list[dict[str, Any]]:
    """Collect papers from Semantic Scholar for multiple queries.

    Deduplicates by arxiv_id, filters by date.
    """
    seen: dict[str, dict[str, Any]] = {}
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    cutoff_year = cutoff.year

    for i, query in enumerate(queries):
        papers = search_papers(query, api_key=api_key)
        for paper in papers:
            aid = paper["arxiv_id"]
            if aid in seen:
                continue
            # Filter by year (SS doesn't give exact dates easily)
            try:
                pub_year = int(paper["published"][:4])
                if pub_year < cutoff_year:
                    continue
            except (ValueError, IndexError):
                pass
            seen[aid] = paper

        # Rate limiting between queries
        if i < len(queries) - 1 and rate_limit > 0:
            time.sleep(rate_limit)

    return list(seen.values())
