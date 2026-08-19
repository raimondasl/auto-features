"""DBLP as a paper source for systems / PL / DB / theory repos.

DBLP is a bibliography (no abstracts), but its publication search API is a clean
keyword-search JSON endpoint — a good fit for RepoRadar's query model. Papers are
returned title-only (``abstract=""``); relevance ranking is therefore title-based
until DOI abstract-backfill lands. Non-arXiv papers get a synthetic ``dblp:<key>`` id.

Two properties to know before enabling this source:

- **DBLP rate-limits hard.** Bursts get connections dropped ("Remote end closed
  connection"); measured, 1 of 6 rapid requests succeeded. Every request therefore
  goes through a process-wide minimum interval, and once refused it needs a long
  cool-off — so a run that sweeps many repos will be slow and may still degrade.
- **DBLP only exposes a publication YEAR.** ``collect_papers`` can therefore only
  filter to ``year >= cutoff_year``, which makes it a poor match for short recency
  windows: with a 14- or 90-day lookback only the *current calendar year* survives,
  and the effective window swings with the date. DBLP is best paired with a wide
  lookback (``rr update --foundational``), where its coverage of non-arXiv
  systems/PL/DB literature is the point.
"""

from __future__ import annotations

import json as json_mod
import logging
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from reporadar.paper_id import dedup_id, doi_key

logger = logging.getLogger(__name__)

DBLP_SEARCH_URL = "https://dblp.org/search/publ/api"

# DBLP drops connections under bursts (measured: 1 of 6 rapid requests succeeded,
# the rest RemoteDisconnected), so every request goes through a process-wide
# minimum interval. Spacing queries only *within* collect_papers is not enough —
# callers that loop over repos (the eval sweeps 12 cases) issue dozens of calls.
_MIN_REQUEST_INTERVAL_S = 1.5
_last_request_at = 0.0


def _throttle() -> None:
    """Block until at least ``_MIN_REQUEST_INTERVAL_S`` since the last request."""
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext | None:
    """TLS context that trusts dblp.org's research-network CA.

    dblp.org's certificate chains to GEANT / Hellenic Academic and Research
    Institutions CA. That root ships in the ``certifi`` bundle but is *not* in the
    Windows system trust store Python's default context uses, so verification
    fails there with CERTIFICATE_VERIFY_FAILED. Prefer certifi when it's available
    (it is, transitively via ``arxiv``→``requests``) and fall back to the default.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - certifi absent or unreadable
        return None


# DBLP CoRR keys mirror arXiv: journals/corr/abs-2401-12345 -> 2401.12345
_CORR_KEY_RE = re.compile(r"corr/abs-(\d{4})-(\d{4,5})")


def _text(value: Any) -> str:
    """Coerce a DBLP field to a single string.

    DBLP's JSON collapses a single repeated XML element to a scalar but expands
    multiple to a LIST, and typed elements become ``{'@type': ..., 'text': ...}``.
    So fields like ``ee``/``url``/``venue`` can be str | list | dict — normalize
    them so they never reach SQLite as a non-bindable list/dict.
    """
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("text", "")
    return str(value) if value else ""


def _arxiv_id(info: dict[str, Any]) -> str | None:
    """Recover the bare arXiv id for a DBLP CoRR (arXiv-mirror) entry, else None.

    Lets arXiv papers found via DBLP dedup against the primary arXiv source
    instead of being stored a second time as a title-only ``dblp:...`` copy.
    """
    doi = _text(info.get("doi")).lower()
    if "arxiv." in doi:
        return doi.split("arxiv.")[-1]
    for link in (_text(info.get("ee")), _text(info.get("url"))):
        if "arxiv.org/abs/" in link:
            return dedup_id(link.split("arxiv.org/abs/")[-1])
    match = _CORR_KEY_RE.search(str(info.get("key") or ""))
    return f"{match.group(1)}.{match.group(2)}" if match else None


def _request_json(url: str, max_retries: int = 3, base_delay: float = 5.0) -> Any | None:
    """GET a JSON endpoint with throttling + retry/backoff; None on failure.

    ``base_delay`` is deliberately longer than the other adapters': once DBLP has
    started refusing, a 2s pause is not enough to get back in.
    """
    headers = {
        "Accept": "application/json",
        # Identify the client rather than sending the default python-urllib agent.
        "User-Agent": "RepoRadar/1.0 (+https://github.com/raimondasl/auto-features)",
    }
    for attempt in range(max_retries):
        try:
            _throttle()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                time.sleep(base_delay * (2**attempt))
                continue
            logger.warning("DBLP API error: %s", exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                logger.warning("DBLP request failed after %d attempts: %s", max_retries, exc)
                return None
    return None


def _authors(info: dict[str, Any]) -> list[str]:
    """Extract author names from DBLP's ``authors.author`` (list | dict | str)."""
    author_field = (info.get("authors") or {}).get("author")
    if author_field is None:
        return []
    items = author_field if isinstance(author_field, list) else [author_field]
    names: list[str] = []
    for item in items:
        name = item.get("text", "") if isinstance(item, dict) else str(item)
        if name:
            names.append(name)
    return names


def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a DBLP search hit to the internal paper dict, or None if unusable."""
    info = hit.get("info") or {}
    title = _text(info.get("title")).rstrip(".")
    key = info.get("key") or info.get("doi") or ""
    if not title or not key:
        return None

    year = info.get("year")
    published = f"{year}-01-01T00:00:00+00:00" if year else datetime.now(UTC).isoformat()
    venue = _text(info.get("venue"))
    common = {
        "title": title,
        "authors": _authors(info),
        "abstract": "",  # DBLP has no abstracts
        "categories": [venue] if venue else [],
        "published": published,
        "updated": None,
    }

    arxiv = _arxiv_id(info)
    if arxiv:
        # A CoRR/arXiv paper: use the real arXiv id + links so it collapses onto
        # the arXiv source instead of duplicating as a title-only DBLP entry.
        return {
            **common,
            "arxiv_id": arxiv,
            "url": f"http://arxiv.org/abs/{arxiv}",
            "pdf_url": f"http://arxiv.org/pdf/{arxiv}",
        }
    return {
        **common,
        # The DOI first (F15): a DBLP hit for a journal or conference paper is the same
        # record OpenAlex and Semantic Scholar return, and `dblp:<key>` is a DBLP handle
        # that agrees with neither. The key remains the fallback for an entry with no DOI.
        "arxiv_id": doi_key(info.get("doi")) or f"dblp:{key}",
        "url": _text(info.get("ee")) or _text(info.get("url")),
        "pdf_url": None,
    }


def search_papers(query: str, limit: int = 30) -> list[dict[str, Any]]:
    """Search DBLP for publications matching *query*. Internal dict format."""
    url = f"{DBLP_SEARCH_URL}?q={urllib.parse.quote(query)}&format=json&h={limit}"
    data = _request_json(url)
    if not data:
        return []
    hit_field = (((data.get("result") or {}).get("hits") or {}).get("hit")) or []
    # A single-result query returns `hit` as one object, not a 1-element list.
    hits = hit_field if isinstance(hit_field, list) else [hit_field]
    results: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        normalized = _normalize_hit(hit)
        if normalized:
            results.append(normalized)
    return results


def collect_papers(
    queries: list[str],
    lookback_days: int = 14,
    limit: int = 30,
    rate_limit: float = 1.0,
) -> list[dict[str, Any]]:
    """Collect DBLP papers for multiple queries; dedup by id, drop years older than
    the lookback window (recency is then handled by the ranker)."""
    cutoff_year = (datetime.now(UTC) - timedelta(days=lookback_days)).year
    seen: dict[str, dict[str, Any]] = {}
    for i, query in enumerate(queries):
        for paper in search_papers(query, limit=limit):
            aid = paper["arxiv_id"]
            if aid in seen:
                continue
            try:
                if int(paper["published"][:4]) < cutoff_year:
                    continue
            except (ValueError, IndexError):
                pass
            seen[aid] = paper
        if i < len(queries) - 1 and rate_limit > 0:
            time.sleep(rate_limit)
    return list(seen.values())
