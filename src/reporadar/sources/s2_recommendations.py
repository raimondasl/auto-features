"""Semantic Scholar Recommendations — a learned recommender seeded by your ratings.

`paper_ratings`/`paper_stars` only nudge heuristic weights today. The free S2
Recommendations API takes positive **and** negative example papers and returns
papers from a production-trained model (the engine behind S2 Research Feeds) —
a learned recommendation channel at zero local ML cost.

Results are merged into the normal candidate pool (``matched_query="recommendation"``)
so RepoRadar's own ranker re-filters them: the API is repo-agnostic and can return
off-topic general-science papers for a CS seed, so local re-scoring is essential.

API notes (live-verified):
- Requires at least one positive seed; a negative-only call returns HTTP 400.
- Seeds must be ids S2 knows: real arXiv ids, or a bare S2 ``paperId`` (which is
  what our ``ss:`` ids wrap, so those seed fine once the prefix is stripped).
  Ids S2 can't resolve make the whole call 400, so other synthetic ids
  (``doi:``, ``dblp:``, ``biorxiv:``, ``oa:``) are filtered out rather than sent.
  S2 does resolve ``DOI:`` seeds, so ``doi:`` ids could be sent as well — untried,
  so not done: an unresolvable seed costs the whole call, not just itself.
- Works keyless (shared pool, throttled — hence the retry/backoff below); an API
  key is optional.
- Draws from a recent pool, so this surfaces new work, not classic literature.
"""

from __future__ import annotations

import json as json_mod
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

from reporadar import s2_rate
from reporadar.paper_id import doi_key, is_arxiv_id

logger = logging.getLogger(__name__)

S2_RECS_URL = "https://api.semanticscholar.org/recommendations/v1/papers/"
# publicationDate gives the real posting date; without it we'd only have `year`,
# and a year-only date makes genuinely recent recommendations look ~6 months
# stale to the recency scorer (which would mute them — defeating the feature).
S2_RECS_FIELDS = "externalIds,title,abstract,authors,year,publicationDate,url,openAccessPdf"

_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
# Our "ss:" ids wrap a bare S2 paperId (a 40-char hex sha), which this endpoint
# accepts directly — so papers discovered *through* recommendations can seed later runs.
_S2_PAPER_ID_RE = re.compile(r"^[0-9a-f]{40}$")


def _seed_ids(arxiv_ids: list[str], max_seeds: int) -> list[str]:
    """Seed ids S2 can resolve: ``ARXIV:<id>`` or a bare S2 paperId.

    Ids S2 cannot resolve are dropped rather than sent — a single unresolvable
    positive seed makes the whole call fail with HTTP 400.
    """
    out: list[str] = []
    for aid in arxiv_ids:
        if aid.startswith("ss:") and _S2_PAPER_ID_RE.match(aid[3:]):
            seed = aid[3:]  # bare S2 paperId is a first-class id here
        elif is_arxiv_id(aid):
            # Strip only a trailing version — an archive name can contain "v"
            # (e.g. solv-int/9502001v1), so never split on the first "v".
            seed = f"ARXIV:{_VERSION_SUFFIX_RE.sub('', aid)}"
        else:
            continue  # doi:/dblp:/biorxiv:/oa:/malformed -> would 400 the whole call
        if seed not in out:
            out.append(seed)
        if len(out) >= max_seeds:
            break
    return out


def _normalize(paper: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an S2 recommendation to the internal paper dict, or None."""
    title = paper.get("title") or ""
    if not title:
        return None

    external = paper.get("externalIds") or {}
    arxiv = external.get("ArXiv")
    pdf_url: str | None
    if arxiv:
        arxiv_id = str(arxiv)
        url = paper.get("url") or f"http://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"
    else:
        # Same rule as the search adapter: the DOI is the cross-source id when there is one
        # (F15), and the S2 handle is the fallback.
        canonical = doi_key(external.get("DOI"))
        paper_id = paper.get("paperId")
        if not canonical and not paper_id:
            return None
        arxiv_id = canonical or f"ss:{paper_id}"
        url = paper.get("url") or ""
        pdf_url = (paper.get("openAccessPdf") or {}).get("url")

    # Prefer the exact publication date; fall back to the year, then to "now", so
    # the recency scorer sees a real date rather than an artificial Jan-1 stamp.
    pub_date = paper.get("publicationDate")
    year = paper.get("year")
    if pub_date:
        published = f"{pub_date}T00:00:00+00:00"
    elif year:
        published = f"{year}-01-01T00:00:00+00:00"
    else:
        published = datetime.now(UTC).isoformat()

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")],
        "abstract": paper.get("abstract") or "",
        "categories": [],  # S2 doesn't return arXiv categories here
        "published": published,
        "updated": None,
        "url": url,
        "pdf_url": pdf_url,
        "matched_query": "recommendation",
    }


def fetch_recommendations(
    positive_ids: list[str],
    negative_ids: list[str] | None = None,
    limit: int = 20,
    max_seeds: int = 50,
    api_key: str | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[dict[str, Any]] | None:
    """Recommend papers from positive (liked) and negative (disliked) examples.

    Returns internal paper dicts; ``[]`` when there are no usable positive seeds
    or the API genuinely returned nothing; and ``None`` when the request failed
    (so the caller can say "unavailable" rather than "no results"). Retries with
    backoff on 429/5xx, since the keyless pool throttles aggressively.
    """
    positives = _seed_ids(positive_ids, max_seeds)
    if not positives:
        # A call without positives is guaranteed to 400 — don't make it.
        return []
    negatives = _seed_ids(negative_ids or [], max_seeds)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    body = json_mod.dumps({"positivePaperIds": positives, "negativePaperIds": negatives}).encode(
        "utf-8"
    )
    url = f"{S2_RECS_URL}?fields={S2_RECS_FIELDS}&limit={max(1, limit)}"

    data: Any = None
    for attempt in range(max_retries):
        try:
            # Shared 1 RPS gate: this endpoint counts against the same per-key budget as
            # search and /paper/batch, and `rr update` can call all three in one run.
            s2_rate.wait_turn()
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json_mod.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                if exc.code == 429:
                    s2_rate.note_throttled()
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
                    continue
                logger.warning("S2 recommendations throttled (HTTP %s); giving up.", exc.code)
                return None
            # 400 = a seed S2 can't resolve; retrying won't help.
            logger.warning("S2 recommendations error %s: %s", exc.code, exc)
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
                continue
            logger.warning("S2 recommendations request failed: %s", exc)
            return None
    if data is None:
        return None

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in data.get("recommendedPapers") or []:
        normalized = _normalize(entry or {})
        if normalized and normalized["arxiv_id"] not in seen:
            seen.add(normalized["arxiv_id"])
            results.append(normalized)
    return results
