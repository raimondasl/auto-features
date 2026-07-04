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
    except Exception:
        return None
    return _normalize(result) if result is not None else None


def resolve_by_title(client: arxiv.Client, title: str) -> dict[str, Any] | None:
    title = title.strip()
    if len(title) < 8:
        return None
    try:
        result = next(client.results(arxiv.Search(query=f'ti:"{title}"', max_results=1)), None)
    except Exception:
        return None
    if result is None:
        return None
    # Guard against loose title matches.
    got = result.title.strip().lower()
    if title.lower() not in got and got not in title.lower():
        return None
    return _normalize(result)


def resolve_references(
    ids: list[str],
    titles: list[str],
    client: arxiv.Client | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Resolve proposed papers to real arXiv metadata.

    Returns (resolved_papers, n_hallucinated). Deduplicates by arxiv_id.
    """
    client = client or arxiv.Client(page_size=25, delay_seconds=3.0, num_retries=2)
    resolved: dict[str, dict[str, Any]] = {}
    hallucinated = 0

    for arxiv_id in ids:
        paper = resolve_by_id(client, arxiv_id)
        if paper is None:
            hallucinated += 1
        else:
            resolved.setdefault(paper["arxiv_id"].split("v")[0], paper)

    for title in titles:
        paper = resolve_by_title(client, title)
        if paper is None:
            hallucinated += 1
        else:
            resolved.setdefault(paper["arxiv_id"].split("v")[0], paper)

    return list(resolved.values()), hallucinated
