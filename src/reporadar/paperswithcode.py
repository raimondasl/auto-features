"""Papers With Code API client for enrichment data (code repos, datasets, tasks)."""

from __future__ import annotations

import json as json_mod
import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

PWC_API_BASE = "https://paperswithcode.com/api/v1"


def _request_json(url: str, max_retries: int = 3, base_delay: float = 1.0) -> Any | None:
    """GET a JSON endpoint with retry and backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                return None
            if exc.code == 429 or exc.code >= 500:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "PwC API error %d (attempt %d/%d). Retrying in %.1fs...",
                    exc.code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("PwC API error: %s", exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "PwC request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.warning("PwC API failed after %d attempts: %s", max_retries, exc)
                return None

    logger.warning("PwC API failed after %d attempts: %s", max_retries, last_exc)
    return None


def fetch_paper_info(arxiv_id: str) -> dict[str, Any] | None:
    """Fetch paper info from Papers With Code by arXiv ID.

    Returns the PwC paper data dict or None if not found / on error.
    """
    # Strip version suffix (e.g., "2401.12345v1" -> "2401.12345")
    base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    url = f"{PWC_API_BASE}/papers/?arxiv_id={base_id}"
    data = _request_json(url)
    if data is None:
        return None

    # The API returns a paginated list; take first result
    results = data.get("results", [])
    if not results:
        return None
    return results[0]


def fetch_enrichment(arxiv_id: str) -> dict[str, Any] | None:
    """Fetch enrichment data (repos, datasets, tasks) for a paper.

    Returns a dict with keys: arxiv_id, pwc_id, has_code, code_urls,
    datasets, tasks. Returns None on failure.
    """
    paper = fetch_paper_info(arxiv_id)
    if paper is None:
        return None

    pwc_id = paper.get("id", "")
    if not pwc_id:
        return None

    # Fetch repositories
    repos_data = _request_json(f"{PWC_API_BASE}/papers/{pwc_id}/repositories/")
    code_urls: list[str] = []
    if repos_data and "results" in repos_data:
        for repo in repos_data["results"]:
            repo_url = repo.get("url", "")
            if repo_url:
                code_urls.append(repo_url)

    # Fetch datasets
    datasets_data = _request_json(f"{PWC_API_BASE}/papers/{pwc_id}/datasets/")
    datasets: list[str] = []
    if datasets_data and "results" in datasets_data:
        for ds in datasets_data["results"]:
            name = ds.get("name", "")
            if name:
                datasets.append(name)

    # Fetch tasks
    tasks_data = _request_json(f"{PWC_API_BASE}/papers/{pwc_id}/tasks/")
    tasks: list[str] = []
    if tasks_data and "results" in tasks_data:
        for task in tasks_data["results"]:
            name = task.get("name", "")
            if name:
                tasks.append(name)

    return {
        "arxiv_id": arxiv_id,
        "pwc_id": pwc_id,
        "has_code": len(code_urls) > 0,
        "code_urls": code_urls,
        "datasets": datasets,
        "tasks": tasks,
    }


def fetch_enrichments_batch(
    arxiv_ids: list[str],
    rate_limit: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Fetch enrichments for multiple papers with rate limiting.

    Returns ``{arxiv_id: enrichment_dict}``. Graceful degradation:
    logs warnings and returns partial results on API failure.
    """
    results: dict[str, dict[str, Any]] = {}
    for i, arxiv_id in enumerate(arxiv_ids):
        try:
            enrichment = fetch_enrichment(arxiv_id)
            if enrichment is not None:
                results[arxiv_id] = enrichment
        except Exception as exc:
            logger.warning("Failed to enrich %s: %s", arxiv_id, exc)

        # Rate limiting between requests (skip after last)
        if i < len(arxiv_ids) - 1 and rate_limit > 0:
            time.sleep(rate_limit)

    return results
