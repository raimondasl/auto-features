"""OpenAlex as a free, no-key-required paper search source."""

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

OA_API_BASE = "https://api.openalex.org"


def _request_json(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any | None:
    """GET a JSON endpoint with retry and backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429 or exc.code >= 500:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "OpenAlex API error %d (attempt %d/%d). Retrying in %.1fs...",
                    exc.code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("OpenAlex API error: %s", exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "OpenAlex request failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.warning("OpenAlex API failed after %d attempts: %s", max_retries, exc)
                return None

    logger.warning("OpenAlex API failed after %d attempts: %s", max_retries, last_exc)
    return None


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct abstract text from OpenAlex's inverted index format.

    OpenAlex stores abstracts as ``{word: [positions]}``. This function
    reverses the index to produce a readable string.
    """
    if not inverted_index:
        return ""

    # Find max position to size the array
    max_pos = 0
    for positions in inverted_index.values():
        for pos in positions:
            if pos > max_pos:
                max_pos = pos

    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word

    return " ".join(w for w in words if w)


def _extract_arxiv_id(work: dict[str, Any]) -> str:
    """Extract arXiv ID from an OpenAlex work, or generate synthetic ID."""
    # Check IDs for arXiv
    ids = work.get("ids", {})
    openalex_id = ids.get("openalex", "") or work.get("id", "")

    # Check DOI for arXiv
    doi = work.get("doi", "") or ids.get("doi", "")
    if doi and "arxiv" in doi.lower():
        # e.g., https://doi.org/10.48550/arXiv.2401.12345
        parts = doi.split("arXiv.")
        if len(parts) > 1:
            return parts[-1]

    # Synthetic ID
    if openalex_id:
        oa_id = openalex_id.replace("https://openalex.org/", "")
        return f"oa:{oa_id}"

    return ""


def _normalize_paper(work: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAlex work to internal paper format."""
    title = work.get("title", "") or work.get("display_name", "")
    if not title:
        return None

    arxiv_id = _extract_arxiv_id(work)
    if not arxiv_id:
        return None

    # Authors
    authors: list[str] = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)

    # Abstract
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    # Publication date
    pub_date = work.get("publication_date", "")
    published = f"{pub_date}T00:00:00+00:00" if pub_date else datetime.now(UTC).isoformat()

    # Categories from primary topic
    categories: list[str] = []
    topic = work.get("primary_topic")
    if topic and topic.get("display_name"):
        categories.append(topic["display_name"])

    # URL
    doi = work.get("doi", "")
    url = doi if doi else work.get("id", "")

    # Open access PDF
    oa = work.get("open_access", {})
    pdf_url = oa.get("oa_url")

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "categories": categories,
        "published": published,
        "updated": None,
        "url": url,
        "pdf_url": pdf_url,
    }


def search_papers(
    query: str,
    limit: int = 50,
    email: str | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for papers matching a query.

    If *email* is provided, uses the polite pool for faster rate limits.
    """
    params: dict[str, str] = {
        "search": query,
        "per_page": str(min(limit, 200)),
        "filter": "type:article",
        "select": (
            "id,doi,title,authorships,abstract_inverted_index,"
            "primary_topic,publication_date,open_access,ids,display_name"
        ),
    }
    if email:
        params["mailto"] = email

    url = f"{OA_API_BASE}/works?{urllib.parse.urlencode(params)}"

    data = _request_json(url)
    if data is None:
        return []

    results: list[dict[str, Any]] = []
    for work in data.get("results", []):
        normalized = _normalize_paper(work)
        if normalized:
            results.append(normalized)

    return results


def collect_papers(
    queries: list[str],
    email: str | None = None,
    lookback_days: int = 14,
    rate_limit: float = 1.0,
) -> list[dict[str, Any]]:
    """Collect papers from OpenAlex for multiple queries.

    Deduplicates by arxiv_id, filters by date.
    """
    seen: dict[str, dict[str, Any]] = {}
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    cutoff_iso = cutoff.strftime("%Y-%m-%d")

    for i, query in enumerate(queries):
        papers = search_papers(query, email=email)
        for paper in papers:
            aid = paper["arxiv_id"]
            if aid in seen:
                continue
            # Filter by publication date
            pub = paper.get("published", "")[:10]
            if pub and pub < cutoff_iso:
                continue
            seen[aid] = paper

        if i < len(queries) - 1 and rate_limit > 0:
            time.sleep(rate_limit)

    return list(seen.values())
