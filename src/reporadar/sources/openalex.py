"""OpenAlex as a free, no-key-required paper search source."""

from __future__ import annotations

import json as json_mod
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

from reporadar.paper_id import doi_key, is_arxiv_id

logger = logging.getLogger(__name__)

OA_API_BASE = "https://api.openalex.org"


def _request_json(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    status: dict[str, int] | None = None,
) -> Any | None:
    """GET a JSON endpoint with retry and backoff.

    ``status`` optionally receives ``{"http": <code>}`` for a NON-retryable HTTP failure.
    ``None`` alone cannot distinguish *OpenAlex would not answer* (429/5xx, retries spent)
    from *OpenAlex has no such record* (404), and `evals/verify.py` turns that difference
    into a verdict about a paper: the first is transient and must never harden, the second
    is an answer and should fall through. Collapsing them is C-32, which this project has
    now paid for once in `citations._s2_batch_post`; the same shape was already sitting
    here. Product callers ignore it and keep the "None means skip" contract.
    """
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
            # OpenAlex answered, and the answer was "no". Recorded, not raised.
            if status is not None:
                status["http"] = exc.code
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


# `http://arxiv.org/abs/2005.00707`, `https://arxiv.org/pdf/1510.04418v2`, and the pre-2007
# form `arxiv.org/abs/cs/0602007`. The captured id is validated by `is_arxiv_id` before use,
# so a URL shape this does not anticipate degrades to "no arXiv id" rather than to a bad one.
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", re.IGNORECASE)


def _arxiv_id_from_locations(work: dict[str, Any]) -> str:
    """An arXiv id from the work's OA locations, if OpenAlex records one.

    A journal article and its arXiv preprint are one paper to a reader and, until this,
    two ids to the pipeline: OpenAlex's `ids` block carries only doi/openalex/pmid/mag, so
    the published version got a `doi:` id and never merged with the arXiv copy already in
    the pool. §39.5 measured what that costs — the same paper shown TWICE in five of the
    five benchmark cases the OpenAlex channel contributed to, against zero in the control.

    `locations` is where OpenAlex does record it. Measured over those five duplicate pairs
    it names an arXiv landing page for **two**; the other three (CHGNet included) list none,
    so this closes part of the defect and the rest would need title matching, which §30 is
    the standing reason not to trust.
    """
    for loc in work.get("locations") or []:
        if not isinstance(loc, dict):
            continue
        for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
            match = _ARXIV_URL_RE.search(str(url or ""))
            if not match:
                continue
            candidate = match.group(1).removesuffix(".pdf")
            if is_arxiv_id(candidate):
                return candidate
    return ""


def _extract_arxiv_id(work: dict[str, Any]) -> str:
    """The arXiv id, else the DOI id, else a synthetic OpenAlex one.

    `oa:W...` is an OpenAlex handle, so the same preprint reached the pool again under
    `ss:` from Semantic Scholar and `biorxiv:` from bioRxiv. Preferring the DOI (F15) gives
    all three the same id. See :func:`reporadar.paper_id.doi_key`.

    An arXiv id beats the DOI where one is known, because the arXiv channel is always on and
    a `doi:` id cannot merge with what it already collected. `sources/semantic_scholar.py`
    has always ordered it that way (`externalIds["ArXiv"]` before `doi_key`); this adapter
    was the one that did not. The digest's link is unaffected — `_normalize_paper` builds it
    from the DOI independently — so this changes which record survives the merge, not what a
    user clicks.
    """
    # Check IDs for arXiv
    ids = work.get("ids", {})
    openalex_id = ids.get("openalex", "") or work.get("id", "")

    # Check DOI for arXiv
    doi = work.get("doi", "") or ids.get("doi", "")
    if doi and "arxiv" in doi.lower():
        # e.g. https://doi.org/10.48550/arXiv.2401.12345 — but OpenAlex normalises DOIs,
        # so the same record can arrive lowercased. The guard above is case-insensitive;
        # splitting case-sensitively meant a lowercase DOI passed the guard, failed the
        # split, and fell through to a synthetic `oa:W...` id — the same paper arXiv had
        # already supplied, now undeduplicable against it.
        parts = re.split(r"arxiv\.", doi, flags=re.IGNORECASE)
        if len(parts) > 1:
            return str(parts[-1])

    from_locations = _arxiv_id_from_locations(work)
    if from_locations:
        return from_locations

    canonical = doi_key(doi)
    if canonical:
        return canonical

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
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for papers matching a query.

    If *api_key* is provided it is sent as the ``api_key`` query parameter
    (OpenAlex requires a key for its full free allowance since 2026-02-13).
    If *email* is provided, uses the legacy polite pool.
    """
    params: dict[str, str] = {
        "search": query,
        "per_page": str(min(limit, 200)),
        # `type:article` alone excludes every preprint, and OpenAlex has counted preprints
        # as a separate type since 2024 — so this source, whose whole purpose is reaching
        # literature arXiv does not carry, was filtering out the preprint servers. Probed
        # 2026-08-19 over six bio and materials queries with a date filter, which is how the
        # pipeline uses it: preprints are **26.7%** of the last-30-day pool and 22.0% of the
        # last-180-day pool, and the venues they bring are bioRxiv (26), arXiv (32),
        # ChemRxiv (9), Research Square (8) and medRxiv. With `sources/biorxiv.py` broken by
        # construction (B1), this is currently the only wired route to bioRxiv at all.
        #
        # Note it substitutes rather than adds: `per_page` is unchanged, so a quarter of the
        # recent pool becomes preprints instead of the pool growing by a quarter.
        "filter": "type:article|preprint",
        "select": (
            "id,doi,title,authorships,abstract_inverted_index,"
            "primary_topic,publication_date,open_access,ids,display_name,locations"
        ),
    }
    if api_key:
        params["api_key"] = api_key
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
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Collect papers from OpenAlex for multiple queries.

    Deduplicates by arxiv_id, filters by date.
    """
    seen: dict[str, dict[str, Any]] = {}
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    cutoff_iso = cutoff.strftime("%Y-%m-%d")

    for i, query in enumerate(queries):
        papers = search_papers(query, email=email, api_key=api_key)
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


def fetch_work_by_doi(doi: str, status: dict[str, int] | None = None) -> dict[str, Any] | None:
    """One OpenAlex work by DOI, raw. ``None`` when absent or unreachable.

    Deliberately returns the raw work rather than a normalised paper. `_normalize_paper`
    cannot serve this caller: it returns ``None`` for any work without an arXiv id, which is
    every paper this function exists to fetch. Rather than loosen a rule the collection path
    depends on, the eval verifier normalises what it needs and reuses
    :func:`reconstruct_abstract`, which is the part with the actual logic in it.

    No ``mailto`` is sent. OpenAlex's polite pool wants an address and the anonymous pool is
    sufficient at this volume, and a user's email is not ours to hand to a third party for
    a rate-limit courtesy.
    """
    key = doi_key(doi).removeprefix("doi:")
    if not key:
        return None
    url = f"{OA_API_BASE}/works/doi:{urllib.parse.quote(key, safe='')}"
    work = _request_json(url, status=status)
    return work if isinstance(work, dict) else None
