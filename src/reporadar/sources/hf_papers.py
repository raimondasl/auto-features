"""Hugging Face Papers enrichment (code repos, linked models, datasets, upvotes).

Replaces the sunset Papers With Code API (paperswithcode.com was shut down in
July 2025 and now redirects to Hugging Face). Uses the free, no-auth Hugging
Face Papers JSON API:

* ``GET /api/papers/{arxiv_id}``            — paper page: upvotes, github repo
* ``GET /api/models?filter=arxiv:{id}``     — models that cite the paper
* ``GET /api/datasets?filter=arxiv:{id}``   — datasets that cite the paper

An optional ``HF_TOKEN`` raises the anonymous rate limit (500 req/5min) to
1000 req/5min but is not required. Enrichment dicts are shape-compatible with
``PaperStore.save_enrichments`` (``arxiv_id``, ``has_code``, ``code_urls``,
``datasets``, ``tasks``, ``models``, ``upvotes``).
"""

from __future__ import annotations

import json as json_mod
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"

# Cap how many linked artifacts we keep per paper to keep digests readable.
MAX_ITEMS = 10


def _request_json(
    url: str,
    token: str | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any | None:
    """GET a JSON endpoint with retry and backoff on transient errors."""
    last_exc: Exception | None = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                return None
            if exc.code == 429 or exc.code >= 500:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "HF API error %d (attempt %d/%d). Retrying in %.1fs...",
                    exc.code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("HF API error: %s", exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "HF request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.warning("HF API failed after %d attempts: %s", max_retries, exc)
                return None

    logger.warning("HF API failed after %d attempts: %s", max_retries, last_exc)
    return None


def _base_arxiv_id(arxiv_id: str) -> str:
    """Strip a version suffix (e.g. ``2401.12345v2`` -> ``2401.12345``)."""
    return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id


def _repo_ids(items: Any) -> list[str]:
    """Extract repo ``id`` strings from a HF list response."""
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            rid = item.get("id") or item.get("modelId")
            if rid:
                ids.append(str(rid))
    return ids[:MAX_ITEMS]


def _code_urls_from_paper(paper: dict[str, Any]) -> list[str]:
    """Build GitHub code URLs from a HF paper object, if it links a repo."""
    repo = paper.get("githubRepo") or paper.get("github")
    if not repo:
        return []
    repo = str(repo)
    if repo.startswith("http"):
        return [repo]
    return [f"https://github.com/{repo.strip('/')}"]


def normalize_upvotes(upvotes_by_id: dict[str, int]) -> dict[str, float]:
    """Log-scale HF upvote counts into ``[0, 1]`` relative to the run's maximum.

    Upvotes are heavy-tailed (a few papers get hundreds, most get none), so a
    linear scale would give everything below the leader a near-zero score. Same
    ``log(1 + v) / log(1 + max)`` treatment as
    :func:`reporadar.citations.normalize_citations`.

    Papers with zero upvotes are **omitted**, not scored 0: zero upvotes usually
    means "HF has no page for this paper", which is an absent signal rather than
    evidence of no interest — and the ranker only counts components that are
    present, so omitting them avoids a silent penalty.
    """
    counted = {k: v for k, v in upvotes_by_id.items() if v and v > 0}
    if not counted:
        return {}
    denom = math.log(1 + max(counted.values()))
    if denom <= 0:  # pragma: no cover - max >= 1 here, so log(1+max) > 0
        return {}
    return {k: math.log(1 + v) / denom for k, v in counted.items()}


def fetch_enrichment(arxiv_id: str, token: str | None = None) -> dict[str, Any] | None:
    """Fetch enrichment data for a single paper from Hugging Face.

    Returns a dict with keys ``arxiv_id``, ``hf_id``, ``has_code``,
    ``code_urls``, ``datasets``, ``tasks``, ``models``, ``upvotes`` — or
    ``None`` when the paper is unknown to HF and has no linked artifacts.
    """
    base_id = _base_arxiv_id(arxiv_id)
    q = urllib.parse.quote(base_id, safe="")

    paper = _request_json(f"{HF_API_BASE}/papers/{q}", token=token)

    models_raw = _request_json(f"{HF_API_BASE}/models?filter=arxiv:{q}&limit={MAX_ITEMS}", token)
    datasets_raw = _request_json(
        f"{HF_API_BASE}/datasets?filter=arxiv:{q}&limit={MAX_ITEMS}", token
    )

    models = _repo_ids(models_raw)
    datasets = _repo_ids(datasets_raw)

    code_urls: list[str] = []
    upvotes = 0
    hf_id = ""
    if isinstance(paper, dict):
        hf_id = str(paper.get("id", base_id))
        upvotes = int(paper.get("upvotes") or 0)
        code_urls = _code_urls_from_paper(paper)

    # Nothing found anywhere — treat as no enrichment (mirrors PwC behavior).
    if paper is None and not models and not datasets:
        return None

    return {
        "arxiv_id": arxiv_id,
        "hf_id": hf_id,
        "has_code": len(code_urls) > 0,
        "code_urls": code_urls,
        "datasets": datasets,
        "tasks": [],  # HF has no "tasks" concept; kept for schema compatibility.
        "models": models,
        "upvotes": upvotes,
    }


def fetch_enrichments_batch(
    arxiv_ids: list[str],
    rate_limit: float = 0.5,
    token: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch enrichments for multiple papers with rate limiting.

    Returns ``{arxiv_id: enrichment_dict}``. Graceful degradation: logs
    warnings and returns partial results on API failure.
    """
    results: dict[str, dict[str, Any]] = {}
    for i, arxiv_id in enumerate(arxiv_ids):
        try:
            enrichment = fetch_enrichment(arxiv_id, token=token)
            if enrichment is not None:
                results[arxiv_id] = enrichment
        except Exception as exc:
            logger.warning("Failed to enrich %s: %s", arxiv_id, exc)

        if i < len(arxiv_ids) - 1 and rate_limit > 0:
            time.sleep(rate_limit)

    return results
