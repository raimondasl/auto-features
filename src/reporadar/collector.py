"""arXiv paper collection — query building and API calls."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import arxiv

from reporadar.config import ArxivConfig, QueriesConfig
from reporadar.profiler import RepoProfile

logger = logging.getLogger(__name__)


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


def collect_papers(
    queries: list[str],
    arxiv_cfg: ArxivConfig,
) -> list[dict[str, Any]]:
    """Execute arXiv queries and return deduplicated paper dicts.

    Deduplication is by arxiv_id (first result wins).
    """
    client = arxiv.Client(
        page_size=arxiv_cfg.max_results_per_query,
        delay_seconds=3.0,
        num_retries=3,
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=arxiv_cfg.lookback_days)
    seen_ids: set[str] = set()
    papers: list[dict[str, Any]] = []

    for query_str in queries:
        logger.info("Querying arXiv: %s", query_str)
        search = arxiv.Search(
            query=query_str,
            max_results=arxiv_cfg.max_results_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        for result in client.results(search):
            # Skip papers older than lookback window
            if result.published.replace(tzinfo=timezone.utc) < cutoff:
                continue

            paper = _result_to_paper(result)
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                paper["matched_query"] = query_str
                papers.append(paper)

    logger.info("Collected %d unique papers from %d queries", len(papers), len(queries))
    return papers
