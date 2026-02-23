"""Citation count lookup via the Semantic Scholar API."""

from __future__ import annotations

import json as json_mod
import logging
import math
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def fetch_citation_counts(
    arxiv_ids: list[str],
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict[str, int]:
    """Fetch citation counts from Semantic Scholar batch endpoint.

    Converts arxiv_ids to Semantic Scholar format (``ARXIV:{id}``) and
    calls the batch paper endpoint.

    Returns ``{arxiv_id: citation_count}`` dict. Returns empty dict on
    API failure (graceful degradation).
    """
    if not arxiv_ids:
        return {}

    # Strip version suffix for Semantic Scholar (e.g. "2401.12345v1" -> "2401.12345")
    clean_ids = {}
    for aid in arxiv_ids:
        base_id = aid.split("v")[0] if "v" in aid else aid
        clean_ids[aid] = f"ARXIV:{base_id}"

    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    payload = json_mod.dumps({"ids": list(clean_ids.values())}).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{url}?fields=citationCount",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data: list[dict[str, Any] | None] = json_mod.loads(resp.read())

            # Build result mapping
            result: dict[str, int] = {}
            original_ids = list(clean_ids.keys())
            for i, entry in enumerate(data):
                if entry is not None and "citationCount" in entry:
                    result[original_ids[i]] = entry["citationCount"]
            return result

        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                # Rate limited
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Semantic Scholar rate limited (attempt %d/%d). Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("Semantic Scholar API error: %s", exc)
            return {}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Semantic Scholar request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "Semantic Scholar API failed after %d attempts: %s",
                    max_retries,
                    exc,
                )
                return {}

    logger.warning("Semantic Scholar API failed after %d attempts: %s", max_retries, last_exc)
    return {}


def normalize_citations(counts: dict[str, int]) -> dict[str, float]:
    """Normalize citation counts to [0, 1] using log scaling.

    Formula: ``log(1 + count) / log(1 + max_count)``
    """
    if not counts:
        return {}

    max_count = max(counts.values())
    if max_count == 0:
        return {k: 0.0 for k in counts}

    denom = math.log(1 + max_count)
    return {k: math.log(1 + v) / denom for k, v in counts.items()}
