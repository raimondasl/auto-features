"""arXiv paper collection — query building and API calls."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import arxiv

from reporadar.config import ArxivConfig, QueriesConfig
from reporadar.profiler import RepoProfile

logger = logging.getLogger(__name__)


class CollectionError(Exception):
    """Raised when arXiv collection fails after exhausting retries."""


def build_queries(
    profile: RepoProfile,
    queries_cfg: QueriesConfig,
    arxiv_cfg: ArxivConfig,
    max_auto_queries: int = 5,
) -> list[str]:
    """Build arXiv query strings from the repo profile and config.

    Strategy:
    1. Use all user-provided seed queries (scoped to configured categories).
    2. Auto-generate queries from the top profile keywords.

    Returns a list of arXiv query strings ready for the API.
    """
    cat_filter = _category_filter(arxiv_cfg.categories)
    queries: list[str] = []

    # Seed queries from config
    for seed in queries_cfg.seed:
        q = f'all:"{seed}"'
        if cat_filter:
            q = f"({q}) AND ({cat_filter})"
        queries.append(q)

    # Auto-generated queries from top keywords
    if profile.keywords:
        top_terms = [term for term, _weight in profile.keywords[:max_auto_queries]]
        for term in top_terms:
            q = f"all:{term}"
            if cat_filter:
                q = f"({q}) AND ({cat_filter})"
            # Skip if it duplicates a seed query
            if q not in queries:
                queries.append(q)

    # Fallback: if no queries at all, search by category only
    if not queries and cat_filter:
        queries.append(cat_filter)

    return queries


def _category_filter(categories: list[str]) -> str:
    """Build an OR-joined category filter string."""
    if not categories:
        return ""
    if len(categories) == 1:
        return f"cat:{categories[0]}"
    parts = [f"cat:{c}" for c in categories]
    return " OR ".join(parts)


def _result_to_paper(result: arxiv.Result) -> dict[str, Any]:
    """Convert an arxiv.Result to our internal paper dict."""
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "abstract": result.summary,
        "categories": result.categories,
        "published": result.published.isoformat(),
        "updated": result.updated.isoformat() if result.updated else None,
        "url": result.entry_id,
        "pdf_url": result.pdf_url,
    }


def _query_with_retry(
    client: arxiv.Client,
    search: arxiv.Search,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[arxiv.Result]:
    """Execute an arXiv query with exponential backoff on transient errors.

    Raises CollectionError if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return list(client.results(search))
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "arXiv query failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise CollectionError(f"arXiv query failed after {max_retries} attempts: {last_exc}")


def collect_papers(
    queries: list[str],
    arxiv_cfg: ArxivConfig,
    on_query_start: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute arXiv queries and return deduplicated paper dicts.

    Deduplication is by arxiv_id (first result wins).

    *on_query_start*, if provided, is called at the start of each query with
    ``(query_index, total_queries, query_string)``.
    """
    client = arxiv.Client(
        page_size=arxiv_cfg.max_results_per_query,
        delay_seconds=3.0,
        num_retries=3,
    )

    cutoff = datetime.now(UTC) - timedelta(days=arxiv_cfg.lookback_days)
    seen_ids: set[str] = set()
    papers: list[dict[str, Any]] = []
    total = len(queries)

    for idx, query_str in enumerate(queries):
        if on_query_start is not None:
            on_query_start(idx, total, query_str)

        logger.info("Querying arXiv: %s", query_str)
        search = arxiv.Search(
            query=query_str,
            max_results=arxiv_cfg.max_results_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results = _query_with_retry(client, search)
        for result in results:
            # Skip papers older than lookback window
            if result.published.replace(tzinfo=UTC) < cutoff:
                continue

            paper = _result_to_paper(result)
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                paper["matched_query"] = query_str
                papers.append(paper)

    logger.info("Collected %d unique papers from %d queries", len(papers), len(queries))
    return papers
