"""Citation lookups via the Semantic Scholar API (counts + reference lists)."""

from __future__ import annotations

import json as json_mod
import logging
import math
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

_S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# Semantic Scholar caps /paper/batch at 500 ids per request; larger id sets are
# chunked so big (e.g. --foundational) runs degrade partially, not to nothing.
_S2_BATCH_LIMIT = 500

# The batch endpoint accepts 500 ids but truncates *nested* items (references, citations)
# at 9,999 across the whole request, allocated greedily in id order. Requests asking for a
# nested field must therefore be chunked by expected nested volume, not by the id limit.
# Measured 2026-08-01 against the live API; see `fetch_references`.
_S2_NESTED_CAP = 9999
_S2_REFERENCE_CHUNK = 50  # ~47 refs/paper observed -> ~2,400 nested, well clear of the cap
_MAX_SPLIT_DEPTH = 4


def _s2_id(arxiv_id: str) -> str:
    """arXiv id → Semantic Scholar id (version-stripped): ``2401.1v2`` → ``ARXIV:2401.1``."""
    base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    return f"ARXIV:{base_id}"


def _chunks(items: list[str], size: int) -> Iterator[tuple[int, list[str]]]:
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def _s2_batch_post(
    s2_ids: list[str],
    fields: str,
    api_key: str | None,
    max_retries: int,
    base_delay: float,
) -> list[Any] | None:
    """POST one batch (<= 500 ids) to the S2 paper/batch endpoint.

    Returns the parsed JSON list on success, or ``None`` on failure (so callers
    can degrade gracefully). Retries with backoff on 429 / transient errors.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    payload = json_mod.dumps({"ids": s2_ids}).encode("utf-8")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                f"{_S2_BATCH_URL}?fields={fields}",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data: list[Any] = json_mod.loads(resp.read())
            return data
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
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
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                logger.warning(
                    "Semantic Scholar request failed after %d attempts: %s", max_retries, exc
                )
                return None
    return None


def fetch_citation_counts(
    arxiv_ids: list[str],
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict[str, int]:
    """Fetch citation counts from the Semantic Scholar batch endpoint.

    Returns ``{arxiv_id: citation_count}``. Ids are chunked into 500-id batches;
    a failed batch is skipped (partial degradation), and a total failure yields
    an empty dict.
    """
    if not arxiv_ids:
        return {}

    original_ids = list(dict.fromkeys(arxiv_ids))  # dedup, preserve order
    s2_ids = [_s2_id(a) for a in original_ids]

    result: dict[str, int] = {}
    for start, chunk in _chunks(s2_ids, _S2_BATCH_LIMIT):
        data = _s2_batch_post(chunk, "citationCount", api_key, max_retries, base_delay)
        if data is None:
            continue
        for i, entry in enumerate(data):
            if entry is not None and "citationCount" in entry:
                result[original_ids[start + i]] = entry["citationCount"]
    return result


def fetch_references(
    arxiv_ids: list[str],
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
    stats: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Fetch each paper's references (the papers it cites) from Semantic Scholar.

    Uses ``fields=references.externalIds``. Returns ``{arxiv_id: [cited_arxiv_id, ...]}``
    where cited ids are version-stripped arXiv ids from each reference's
    ``externalIds.ArXiv``. A failed batch is skipped; a total failure yields an empty dict.

    **Chunk size is bounded by a nested-item cap, not by the id limit.** The batch endpoint
    accepts 500 ids but truncates *nested* results at 9,999 across the whole request, filled
    greedily in id order — so the first few papers consume the budget and every later paper
    comes back with an empty ``references`` array, HTTP 200, no error. Measured on 300 real
    ids: one request returned exactly 9,999 references with **85 of 300 papers blank**; the
    same ids in chunks of 50 returned 14,041 with 8 blank. At the old 500-id chunking this
    feature silently lost ~29% of its edges on any ordinary run.

    A fixed smaller chunk is not enough on its own, because reference density varies by
    corpus — so a chunk that comes back pinned at the cap is split and retried.

    Pass *stats* to learn whether an empty result means "these papers cite nothing on arXiv"
    or "Semantic Scholar did not answer". It is filled with ``requests`` / ``failed`` /
    ``truncated`` counts. Callers that judge a signal by its output — the Tier S harness
    concludes "no signal" from an empty dict — cannot otherwise tell a real negative from an
    outage, and a rate-limited run then reads as evidence about the feature.
    """
    if not arxiv_ids:
        return {}

    original_ids = list(dict.fromkeys(arxiv_ids))
    s2_ids = [_s2_id(a) for a in original_ids]

    result: dict[str, list[str]] = {}
    counts = {"requests": 0, "failed": 0, "truncated": 0}

    def _collect(start: int, chunk: list[str], depth: int = 0) -> None:
        counts["requests"] += 1
        data = _s2_batch_post(chunk, "references.externalIds", api_key, max_retries, base_delay)
        if data is None:
            counts["failed"] += 1
            return
        nested = sum(len((e or {}).get("references") or []) for e in data)
        if nested >= _S2_NESTED_CAP and len(chunk) > 1 and depth < _MAX_SPLIT_DEPTH:
            # Pinned at the cap: the tail of this chunk was silently blanked. Split and
            # retry rather than accept a truncated answer we cannot distinguish from
            # "these papers genuinely have no arXiv-indexed references".
            logger.info(
                "S2 returned %d nested references for %d ids (at the cap) — splitting",
                nested,
                len(chunk),
            )
            mid = len(chunk) // 2
            _collect(start, chunk[:mid], depth + 1)
            _collect(start + mid, chunk[mid:], depth + 1)
            return
        if nested >= _S2_NESTED_CAP:
            counts["truncated"] += 1
        for i, entry in enumerate(data):
            if entry is None:
                continue
            cited: list[str] = []
            for ref in entry.get("references") or []:
                ext = (ref or {}).get("externalIds") or {}
                arxiv = ext.get("ArXiv")
                if arxiv:
                    cited.append(str(arxiv).split("v")[0])
            if cited:
                result[original_ids[start + i]] = cited

    for start, chunk in _chunks(s2_ids, _S2_REFERENCE_CHUNK):
        _collect(start, chunk)
    if stats is not None:
        stats.update(counts)
    return result


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
