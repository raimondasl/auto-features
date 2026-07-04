"""Hallucination guard: resolve proposed papers against the real arXiv.

LLM baselines (and sometimes RepoRadar's non-arXiv sources) can reference papers
that don't exist or misquote IDs. Before judging, every proposed paper is
resolved to real arXiv metadata; anything that can't be resolved is counted as a
hallucination and scored 0, so the comparison stays honest.
"""

from __future__ import annotations

import re
from typing import Any

import arxiv


class ArxivUnavailable(Exception):
    """Raised when an arXiv lookup *errors* (network/HTTP), as opposed to
    succeeding with no match. A lookup error must NOT be counted as a
    hallucination — otherwise an arXiv outage silently turns every real
    baseline paper into an invented one and corrupts the comparison."""


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


# arXiv IDs: new style 2401.12345 (optionally vN), old style hep-th/9901001.
_ID_RE = re.compile(
    r"(?:arxiv[:\s/]*)?(\d{4}\.\d{4,5}(?:v\d+)?)|([a-z-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)


def extract_arxiv_ids(text: str) -> list[str]:
    """Pull arXiv IDs out of free text (e.g. a baseline's answer)."""
    ids: list[str] = []
    for m in _ID_RE.finditer(text):
        rid = m.group(1) or m.group(2)
        if rid and rid not in ids:
            ids.append(rid)
    return ids


def _normalize(result: arxiv.Result) -> dict[str, Any]:
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title.strip().replace("\n", " "),
        "abstract": result.summary.strip().replace("\n", " "),
        "categories": list(result.categories),
        "published": result.published.isoformat() if result.published else "",
        "url": result.entry_id,
    }


def resolve_by_id(client: arxiv.Client, arxiv_id: str) -> dict[str, Any] | None:
    base = arxiv_id.split("v")[0]
    try:
        result = next(client.results(arxiv.Search(id_list=[base])), None)
    except arxiv.UnexpectedEmptyPageError:
        return None  # genuinely no such paper (empty feed)
    except Exception as exc:  # noqa: BLE001 — network/HTTP: lookup failed, NOT "not found"
        raise ArxivUnavailable(f"id lookup failed for {base}: {exc}") from exc
    return _normalize(result) if result is not None else None


def resolve_by_title(client: arxiv.Client, title: str) -> dict[str, Any] | None:
    title = title.strip()
    if len(title) < 8:
        return None
    try:
        # Fetch a few and require a near-exact normalized title match, so a
        # superstring hit ("<title>: A Survey") is not accepted as the paper.
        results = []
        for r in client.results(arxiv.Search(query=f'ti:"{title}"', max_results=5)):
            results.append(r)
            if len(results) >= 5:
                break
    except arxiv.UnexpectedEmptyPageError:
        return None
    except Exception as exc:  # noqa: BLE001 — network/HTTP: lookup failed, NOT "not found"
        raise ArxivUnavailable(f"title lookup failed: {exc}") from exc

    want = _norm_title(title)
    for r in results:
        if _norm_title(r.title) == want:
            return _normalize(r)
    return None


def resolve_references(
    ids: list[str],
    titles: list[str],
    client: arxiv.Client | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Resolve proposed papers to real arXiv metadata.

    Returns (resolved_papers, n_hallucinated, n_lookup_failed). A hallucination
    is a *successful* lookup that found no matching paper; a lookup failure is an
    arXiv network/HTTP error (counted separately so the caller can mark the run
    unverified instead of blaming the baseline). Deduplicates by arxiv_id.
    """
    client = client or arxiv.Client(page_size=25, delay_seconds=3.0, num_retries=2)
    resolved: dict[str, dict[str, Any]] = {}
    hallucinated = 0
    lookup_failed = 0

    def _add(paper: dict[str, Any] | None) -> None:
        nonlocal hallucinated
        if paper is None:
            hallucinated += 1
        else:
            resolved.setdefault(paper["arxiv_id"].split("v")[0], paper)

    for arxiv_id in ids:
        try:
            _add(resolve_by_id(client, arxiv_id))
        except ArxivUnavailable:
            lookup_failed += 1

    for title in titles:
        try:
            _add(resolve_by_title(client, title))
        except ArxivUnavailable:
            lookup_failed += 1

    return list(resolved.values()), hallucinated, lookup_failed
